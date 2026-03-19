"""Executor — internal queue, dispatcher thread, worker pool.

The Executor owns three things:
  1. An internal queue — the handoff point between the router thread / timer
     thread (producers) and the worker threads (consumers).
  2. A dispatcher thread — drains the queue and submits jobs to the pool.
  3. A worker thread pool — runs the handler functions.

The internal queue is the natural extension point for scheduling policy.
Currently FIFO (queue.Queue). To add priority, per-spec concurrency limits,
or backpressure, replace or wrap the queue here — nothing else needs to change.

After each handler returns, the dispatcher calls job.reply_fn(result) if set,
handling SERVICE replies without the handler knowing about query mechanics.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .task import Job


class Executor:
    def __init__(self) -> None:
        # FIFO queue — replace with PriorityQueue for priority scheduling
        self._queue: queue.Queue['Job'] = queue.Queue()
        self._pool: ThreadPoolExecutor | None = None
        self._dispatcher: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dispatch: Callable[['Job'], Any] | None = None

    def enqueue(self, job: 'Job') -> None:
        """Called by the router thread and timer thread to submit a job."""
        self._queue.put(job)

    def start(self, dispatch: Callable[['Job'], Any], n_workers: int = 4) -> None:
        """Start the dispatcher thread and worker pool.

        dispatch — callable provided by AciesApp that runs a job:
            lambda job: job.spec.fn(ctx, job.msg)
        Keeping dispatch as a plain callable means Executor has no knowledge
        of AciesContext, breaking the context → router → executor → context cycle.
        """
        # TODO: Phase 2 — store dispatch, start pool and dispatcher thread
        ...

    def stop(self) -> None:
        # TODO: Phase 2 — signal dispatcher, drain queue, shut down pool
        ...

    def _dispatch_loop(self) -> None:
        # TODO: Phase 2 — drain self._queue, submit each job to self._pool
        ...

    def _run_job(self, job: 'Job') -> None:
        # TODO: Phase 2 — call self._dispatch(job), then job.reply_fn(result)
        ...
