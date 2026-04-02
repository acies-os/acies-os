import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger()


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
        data = defaultdict(dict)

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
