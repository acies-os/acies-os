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
from typing import Callable, TypeAlias

from ._concurrency import SENTINEL, Sentinel
from .task import Job

# Injected by AciesApp; responsible for decoding, calling the handler,
# and sending the reply for ServiceSpec jobs.
Dispatcher: TypeAlias = Callable[[Job], None]


# Sentinel tuple (deadline, created_at, sentinel) sorts last
_SENTINEL_ENTRY = (float('inf'), float('inf'), SENTINEL)


class Executor:
    def __init__(self) -> None:
        # PriorityQueue ordered by (deadline, created_at).
        # deadline=0.0 (default) degrades to FIFO via created_at.
        # Set deadline to a future monotonic timestamp to enable EDF scheduling.
        self._queue: queue.PriorityQueue[tuple[float, float, Job | Sentinel]] = queue.PriorityQueue()
        self._pool: ThreadPoolExecutor | None = None
        self._dispatcher: threading.Thread | None = None
        self._dispatch: Dispatcher | None = None

    def enqueue(self, job: Job) -> None:
        """Called by the router thread and timer thread to submit a job."""
        self._queue.put((job.deadline, job.created_at, job))

    def start(self, dispatch: Dispatcher, n_workers: int = 4) -> None:
        """Start the dispatcher thread and worker pool.

        dispatch — callable provided by AciesApp that runs a job. It is
        responsible for decoding job.raw, calling the handler, and — for
        ServiceSpec jobs — encoding the result and calling job.reply_fn.
        Keeping dispatch as a plain callable means Executor has no knowledge
        of AciesContext or message encoding, breaking any circular dependency.
        """
        self._dispatch = dispatch
        self._pool = ThreadPoolExecutor(max_workers=n_workers)
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name='executor-dispatcher', daemon=True)
        self._dispatcher.start()

    def stop(self) -> None:
        """Drain the queue, then shut down. All enqueued jobs will complete."""
        self._queue.put(_SENTINEL_ENTRY)
        if self._dispatcher:
            self._dispatcher.join()
        if self._pool:
            self._pool.shutdown(wait=True)

    def abort(self) -> None:
        """Stop immediately, discarding queued-but-not-yet-dispatched jobs.

        Jobs already running in worker threads will complete — Python threads
        cannot be forcibly killed. cancel_futures=True cancels pool futures
        that haven't started yet (Python 3.9+).

        There is a small unavoidable race: if the dispatcher has already
        dequeued a job but not yet submitted it to the pool when abort() drains
        the queue, that job will still be submitted.
        """
        while True:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(_SENTINEL_ENTRY)
        if self._dispatcher:
            self._dispatcher.join()
        if self._pool:
            self._pool.shutdown(cancel_futures=True, wait=False)

    def _dispatch_loop(self) -> None:
        assert self._pool is not None, '_dispatch_loop started before pool was initialized'
        while True:
            _, _, item = self._queue.get()
            if item is SENTINEL:
                break
            assert isinstance(item, Job), f'Expected Job, got {type(item)}'
            _ = self._pool.submit(self._run_job, item)

    def _run_job(self, job: Job) -> None:
        assert self._dispatch is not None, '_run_job called before dispatch was initialized'
        self._dispatch(job)
