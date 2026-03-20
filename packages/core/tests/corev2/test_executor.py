import threading
from typing import Any, Callable

from acies.corev2.executor import Executor
from acies.corev2.task import Job, ScheduleSpec

_SPEC = ScheduleSpec(name='test', fn=lambda ctx: None, interval=1.0)


def _make_executor(dispatch: Callable[[Job], Any], n_workers: int = 1):
    ex = Executor()
    ex.start(dispatch, n_workers=n_workers)
    return ex


def test_handler_runs_in_worker_thread():
    worker_thread = []
    done = threading.Event()

    def dispatch(job: Job):
        worker_thread.append(threading.current_thread())
        done.set()
        return None

    ex = _make_executor(dispatch)
    ex.enqueue(Job(spec=_SPEC, msg=None))
    assert done.wait(timeout=2.0)
    ex.stop()

    assert worker_thread[0] is not threading.main_thread()


def test_reply_fn_called_with_return_value():
    results = []
    done = threading.Event()

    def dispatch(job):
        return 42

    def reply_fn(result):
        results.append(result)
        done.set()

    ex = _make_executor(dispatch)
    ex.enqueue(Job(spec=_SPEC, msg=None, reply_fn=reply_fn))
    assert done.wait(timeout=2.0)
    ex.stop()

    assert results == [42]


def test_fifo_ordering_single_worker():
    order = []
    n = 5
    all_done = threading.Event()

    def dispatch(job):
        order.append(job.msg)
        if len(order) == n:
            all_done.set()
        return None

    ex = _make_executor(dispatch, n_workers=1)
    for i in range(n):
        ex.enqueue(Job(spec=_SPEC, msg=i))
    assert all_done.wait(timeout=2.0)
    ex.stop()

    assert order == list(range(n))


def test_stop_drains_queue():
    results = []
    n = 5

    def dispatch(job):
        results.append(job.msg)
        return None

    ex = _make_executor(dispatch, n_workers=1)
    for i in range(n):
        ex.enqueue(Job(spec=_SPEC, msg=i))
    ex.stop()

    assert len(results) == n


def test_abort_discards_queued_jobs():
    dispatched = []
    first_started = threading.Event()
    hold = threading.Event()

    def dispatch(job):
        # Block the first job so the queue fills up behind it
        if job.msg == 0:
            dispatched.append(job.msg)
            first_started.set()
            hold.wait()
        else:
            dispatched.append(job.msg)
        return None

    ex = _make_executor(dispatch, n_workers=1)
    for i in range(5):
        ex.enqueue(Job(spec=_SPEC, msg=i))

    # Wait until job 0 is running, then abort while 1–4 are still queued
    assert first_started.wait(timeout=2.0)
    ex.abort()
    hold.set()  # let job 0 finish (already running, can't be cancelled)

    # Only job 0 should have run; 1–4 were discarded
    assert dispatched == [0]
