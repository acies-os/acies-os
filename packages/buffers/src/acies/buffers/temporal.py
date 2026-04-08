import bisect
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

logger = logging.getLogger()

Timestamp: TypeAlias = int  # nanoseconds since Unix epoch


@dataclass
class TemporalBuffer:
    """Buffer to store data, indexed by topic and timestamp."""

    size: int
    # _data[topic][timestamp] = Value
    _data: dict[str, dict[int, Any]] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _timestamps: Counter[int] = field(default_factory=Counter)

    def add(self, topic: str, timestamp: int, value: Any):
        self._data[topic][timestamp] = value
        self._timestamps[timestamp] += 1
        self._check_size(topic)

    def _check_size(self, topic: str):
        while len(self._data[topic]) > self.size:
            t = min(self._data[topic].keys())
            del self._data[topic][t]
            self._timestamps[t] -= 1
            if self._timestamps[t] == 0:
                del self._timestamps[t]

    def pop(self, keys: list[str], n: int) -> dict[str, dict[int, Any]]:
        """Get n-second of messages for all keys."""
        data: defaultdict[str, dict[int, Any]] = defaultdict(dict)

        # start from the oldest timestamp
        for timestamp in sorted(self._timestamps):
            # find a timestamp that all keys have n samples from [timestamp-n+1, timestamp]
            if all(timestamp - i in self._data[k] for k in keys for i in range(n)):
                for k in keys:
                    for i in reversed(range(n)):
                        t = timestamp - i
                        msg = self._data[k].pop(t)

                        self._timestamps[t] -= 1
                        if self._timestamps[t] == 0:
                            del self._timestamps[t]

                        data[k][t] = msg
                return dict(data)
        raise ValueError('not enough data')

    def clear(self) -> None:
        """Remove all entries."""
        self._data.clear()
        self._timestamps.clear()


@dataclass
class TimeWindow:
    """A map where each key holds a time-sorted list of (timestamp, value) pairs.

    Entries older than ``window_ns`` nanoseconds are pruned on access.
    Timestamps are integers in nanoseconds since epoch (``Timestamp``).

    By default the clock is wall time (``time.time_ns``). Pass
    ``data_clock=True`` to use the latest inserted timestamp as the clock
    instead, which is correct for replayed data whose timestamps are
    historical rather than current wall time.

    Example::

        buf = TimeWindow(window_ns=30 * 1_000_000_000)
        buf.add('rs1/geo', ts, predictions)
        buf.get('rs1/geo')            # [(ts, val), ...]
        buf.nearest('gps/suv', ts)    # (ts, val) closest to ts
        buf.range('rs1/geo', t0, t1)  # entries in [t0, t1]
        buf.pop_aligned(['rs1/geo', 'rs1/mic'], n=5)  # 5 consecutive seconds from all keys
    """

    window_ns: int
    data_clock: bool = False
    _data: dict[str, list[tuple[Timestamp, Any]]] = field(default_factory=dict, repr=False)
    _latest_ts: Timestamp = field(default=0, repr=False)
    _now_fn: Callable[[], Timestamp] | None = field(default=None, repr=False)

    def _now(self) -> Timestamp:
        if self._now_fn is not None:
            return self._now_fn()
        if self.data_clock:
            return self._latest_ts
        return time.time_ns()

    def _prune(self, key: str) -> None:
        """Remove entries older than the window for a single key."""
        entries = self._data.get(key)
        if not entries:
            return
        cutoff = self._now() - self.window_ns
        # Entries with timestamp >= cutoff are kept (inclusive boundary).
        i = bisect.bisect_left(entries, (cutoff,))
        if i > 0:
            del entries[:i]

    def add(self, key: str, timestamp: Timestamp, value: Any) -> None:
        """Insert an entry in sorted order. Safe with out-of-order timestamps."""
        if timestamp > self._latest_ts:
            self._latest_ts = timestamp
        if key not in self._data:
            self._data[key] = []
        bisect.insort(self._data[key], (timestamp, value))
        self._prune(key)

    def get(self, key: str) -> list[tuple[Timestamp, Any]]:
        """Return all entries for a key within the window."""
        self._prune(key)
        return list(self._data.get(key, []))

    def latest(self, key: str) -> tuple[Timestamp, Any] | None:
        """Return the most recent entry for a key, or None."""
        self._prune(key)
        entries = self._data.get(key)
        if not entries:
            return None
        return entries[-1]

    def nearest(self, key: str, timestamp: Timestamp) -> tuple[Timestamp, Any] | None:
        """Return the entry closest to the given timestamp, or None."""
        self._prune(key)
        entries = self._data.get(key)
        if not entries:
            return None
        i = bisect.bisect_left(entries, (timestamp,))
        candidates: list[tuple[Timestamp, Any]] = []
        if i < len(entries):
            candidates.append(entries[i])
        if i > 0:
            candidates.append(entries[i - 1])
        return min(candidates, key=lambda e: abs(e[0] - timestamp))

    def range(self, key: str, t_start: Timestamp, t_end: Timestamp) -> list[tuple[Timestamp, Any]]:
        """Return entries for a key in the time range [t_start, t_end]."""
        self._prune(key)
        entries = self._data.get(key)
        if not entries:
            return []
        lo = bisect.bisect_left(entries, (t_start,))
        # t_end + 1 as a tuple-first-element finds the insertion point after
        # all entries at t_end, regardless of the value component.
        hi = bisect.bisect_left(entries, (t_end + 1,))
        return entries[lo:hi]

    def pop_aligned(
        self,
        keys: list[str],
        n: int,
        step_ns: int = 1_000_000_000,
    ) -> dict[str, list[tuple[Timestamp, Any]]]:
        """Consume and return n consecutive step-aligned windows for all keys.

        Groups each key's nanosecond entries into buckets of size step_ns,
        finds the oldest run of n consecutive buckets present in every key,
        removes all matching entries from the buffer, and returns them as
        {key: [(ts_ns, value), ...]} sorted by timestamp.

        Raises ValueError with 'no common timestamps' if the keys share no
        bucket, or 'no aligned window' if no run of n consecutive buckets
        exists in the intersection.
        """
        if not keys:
            raise ValueError('no keys provided')

        for key in keys:
            self._prune(key)

        # --- phase 1: bucket ids present in each key ---
        def _bids(key: str) -> set[int]:
            counts: dict[int, int] = {}
            for ts_ns, _ in self._data.get(key, []):
                bid = ts_ns // step_ns
                counts[bid] = counts.get(bid, 0) + 1
            for bid, count in counts.items():
                if count > 1:
                    logger.warning(
                        'pop_aligned: key %r has %d entries in bucket %d (step_ns=%d)', key, bid, count, step_ns
                    )
            return set(counts)

        common = _bids(keys[0])
        for key in keys[1:]:
            common &= _bids(key)

        if not common:
            raise ValueError('no common timestamps across all keys')

        # --- phase 2: oldest n consecutive bucket ids ---
        bids = sorted(common)
        start = next(
            (bids[i] for i in range(len(bids) - n + 1) if bids[i + n - 1] - bids[i] == n - 1),
            None,
        )
        if start is None:
            raise ValueError('no aligned window of length n in common timestamps')

        window = set(range(start, start + n))

        # --- phase 3: partition entries, consume window ---
        result: dict[str, list[tuple[Timestamp, Any]]] = {}
        for key in keys:
            keep: list[tuple[Timestamp, Any]] = []
            out: list[tuple[Timestamp, Any]] = []
            for entry in self._data.get(key, []):
                (out if entry[0] // step_ns in window else keep).append(entry)
            self._data[key] = keep
            result[key] = out
        return result

    def keys(self) -> list[str]:
        """Return keys that have at least one entry within the window."""
        result: list[str] = []
        for k in self._data:
            self._prune(k)
            if self._data[k]:
                result.append(k)
        return result

    def __len__(self) -> int:
        """Total number of entries across all keys."""
        return sum(len(v) for v in self._data.values())

    def __contains__(self, key: str) -> bool:
        self._prune(key)
        entries = self._data.get(key)
        return bool(entries)
