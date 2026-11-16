import threading
from typing import Callable

import msgspec

from acies.core.executor import Executor
from acies.core.task import Job, ScheduleSpec

_SPEC = ScheduleSpec(name='test', fn=lambda ctx: None, interval=1.0)


def _make_executor(dispatch: Callable[[Job], None], n_workers: int = 1):
    ex = Executor()
    ex.start(dispatch, n_workers=n_workers)
    return ex


def test_handler_runs_in_worker_thread():
    worker_thread = []
    done = threading.Event()

    def dispatch(job: Job):
        worker_thread.append(threading.current_thread())
        done.set()

    ex = _make_executor(dispatch)
    ex.enqueue(Job(spec=_SPEC, raw=None))
    assert done.wait(timeout=2.0)
    ex.stop()

    assert worker_thread[0] is not threading.main_thread()


def test_dispatch_receives_reply_fn():
    """Executor passes reply_fn in the Job intact; dispatch can invoke it."""
    received: list[bytes] = []
    done = threading.Event()

    def reply_fn(b: bytes) -> None:
        received.append(b)
        done.set()

    def dispatch(job: Job):
        if job.reply_fn is not None:
            job.reply_fn(b'reply')

    ex = _make_executor(dispatch)
    ex.enqueue(Job(spec=_SPEC, raw=None, reply_fn=reply_fn))
    assert done.wait(timeout=2.0)
    ex.stop()

    assert received == [b'reply']


def test_fifo_ordering_single_worker():
    order = []
    n = 5
    all_done = threading.Event()

    def dispatch(job: Job):
        order.append(msgspec.msgpack.decode(job.raw))
        if len(order) == n:
            all_done.set()

    ex = _make_executor(dispatch, n_workers=1)
    for i in range(n):
        ex.enqueue(Job(spec=_SPEC, raw=msgspec.msgpack.encode(i)))
    assert all_done.wait(timeout=2.0)
    ex.stop()

    assert order == list(range(n))


def test_stop_drains_queue():
    results = []
    n = 5

    def dispatch(job: Job):
        results.append(msgspec.msgpack.decode(job.raw))

    ex = _make_executor(dispatch, n_workers=1)
    for i in range(n):
        ex.enqueue(Job(spec=_SPEC, raw=msgspec.msgpack.encode(i)))
    ex.stop()

    assert len(results) == n


def test_abort_discards_queued_jobs():
    dispatched = []
    first_started = threading.Event()
    hold = threading.Event()

    def dispatch(job: Job):
        # Block the first job so the queue fills up behind it
        value = msgspec.msgpack.decode(job.raw)
        if value == 0:
            dispatched.append(value)
            first_started.set()
            hold.wait()
        else:
            dispatched.append(value)

    ex = _make_executor(dispatch, n_workers=1)
    for i in range(5):
        ex.enqueue(Job(spec=_SPEC, raw=msgspec.msgpack.encode(i)))

    # Wait until job 0 is running, then abort while 1–4 are still queued
    assert first_started.wait(timeout=2.0)
    ex.abort()
    hold.set()  # let job 0 finish (already running, can't be cancelled)

    # Only job 0 should have run; 1–4 were discarded
    assert dispatched == [0]
