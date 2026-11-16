"""Rolling operation timings for classifier nodes.

Records how long model loads, weight reloads, and inference calls take, keeps
the last ``window`` samples per operation, and summarises them. Nothing is
published; nodes keep a ``Timings`` in app state (``ctx.app['timings']``), log
the summary periodically and on shutdown, and keep the latest summary in
``sys/timing`` (``record_timing``) so a controller can read it with a kv get.

Usage::

    timings = Timings()
    with timings.measure('reload'):
        model.reload(weight)
    timings.record('infer', infer_ms)
    timings.summary()  # {'reload': {'count': 1, 'last_ms': ..., ...}, ...}
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from acies.core import AciesContext

# how often nodes log the summary
LOG_INTERVAL_S = 60.0


class Timings:
    """Thread-safe rolling durations in milliseconds, per operation name."""

    def __init__(self, window: int = 600) -> None:
        self._samples: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._count: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(self, op: str, ms: float) -> None:
        with self._lock:
            self._samples[op].append(ms)
            self._count[op] += 1

    @contextmanager
    def measure(self, op: str) -> Generator[None]:
        """Record the wall time of the ``with`` block under ``op`` (also when it raises)."""
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record(op, (time.perf_counter_ns() - t0) / 1_000_000)

    def reset(self, op: str | None = None) -> None:
        """Drop the samples and the count of ``op``, or of every op when ``op`` is None.

        Reset steady-state ops (e.g. ``infer``) when the model changes. Event
        ops (``load``, ``reload``) are history; keep them.
        """
        with self._lock:
            if op is None:
                self._samples.clear()
                self._count.clear()
            else:
                self._samples.pop(op, None)
                self._count.pop(op, None)

    def last(self, op: str) -> float | None:
        with self._lock:
            samples = self._samples.get(op)
            return samples[-1] if samples else None

    def summary(self) -> dict[str, dict[str, Any]]:
        """``{op: {count, last_ms, mean_ms, p50_ms, p95_ms, max_ms}}`` over the rolling window."""
        with self._lock:
            snapshot = {op: (list(s), self._count[op]) for op, s in self._samples.items() if s}
        out: dict[str, dict[str, Any]] = {}
        for op, (samples, count) in sorted(snapshot.items()):
            arr = np.asarray(samples)
            out[op] = {
                'count': count,
                'last_ms': round(samples[-1], 2),
                'mean_ms': round(float(arr.mean()), 2),
                'p50_ms': round(float(np.percentile(arr, 50)), 2),
                'p95_ms': round(float(np.percentile(arr, 95)), 2),
                'max_ms': round(float(arr.max()), 2),
            }
        return out

    def format(self) -> str:
        """One-line summary for logs, e.g. ``infer n=60 p50=41.2 p95=48.0 max=52.3ms | reload ...``."""
        parts = [
            f'{op} n={s["count"]} p50={s["p50_ms"]:.1f} p95={s["p95_ms"]:.1f} max={s["max_ms"]:.1f}ms'
            for op, s in self.summary().items()
        ]
        return ' | '.join(parts) or 'no samples'


def record_timing(ctx: AciesContext, timings: Timings) -> None:
    """Store the current summary in ``sys/timing`` as ``{'time_ns': ..., 'ops': summary}``.

    ``time_ns`` tells a reader how old the summary is. The whole dict is replaced at once, so a
    concurrent ``kv get`` never sees a partial update.
    """
    ctx.app.config.setdefault('sys', {})['timing'] = {'time_ns': ctx.now(), 'ops': timings.summary()}
