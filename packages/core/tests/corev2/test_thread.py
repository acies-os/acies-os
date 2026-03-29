"""Tests for @app.thread: startup ordering, shutdown ordering, crash handling."""

import threading
from contextlib import contextmanager

from acies.corev2.app import AciesApp
from acies.corev2.context import AciesContext
from acies.corev2.router import Router
from acies.corev2.transport import LocalTransport

# ---------------------------------- helpers ----------------------------------


def _make_app() -> tuple[AciesApp, threading.Event]:
    router = Router()
    router.add_transport(LocalTransport())
    app = AciesApp('test-app', 'test-host', router=router)

    ready = threading.Event()

    @app.on_startup
    def _set_ready(ctx: AciesContext) -> None:
        ready.set()

    return app, ready


@contextmanager
def running(app: AciesApp, ready: threading.Event, timeout: float = 2.0):
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    assert ready.wait(timeout=timeout), 'app did not become ready in time'
    try:
        yield t
    finally:
        app.stop()
        t.join(timeout=timeout)
        assert not t.is_alive(), 'app.run() did not return after stop()'


# ----------------------------------- tests -----------------------------------


def test_thread_starts_and_receives_stop():
    """Thread starts, receives the stop event, and exits cleanly."""
    app, ready = _make_app()

    started = threading.Event()
    stopped = threading.Event()

    @app.thread
    def worker(ctx: AciesContext, stop: threading.Event) -> None:
        started.set()
        _ = stop.wait()
        stopped.set()

    with running(app, ready=ready):
        assert started.wait(timeout=2.0), 'thread never started'

    assert stopped.is_set(), 'thread did not observe stop event'


def test_thread_starts_after_startup_hook():
    """Thread is started after startup hooks, so ctx.app.data is populated."""
    app, ready = _make_app()

    saw_value: list[object] = []
    thread_started = threading.Event()

    @app.on_startup
    def setup(ctx: AciesContext) -> None:
        ctx.app.data['initialized'] = True

    @app.thread
    def worker(ctx: AciesContext, stop: threading.Event) -> None:
        saw_value.append(ctx.app.data.get('initialized'))
        thread_started.set()
        _ = stop.wait()

    with running(app, ready=ready):
        assert thread_started.wait(timeout=2.0), 'thread never started'

    assert saw_value == [True], 'thread did not see state set by startup hook'


def test_thread_crash_triggers_shutdown():
    """An unhandled exception in a thread calls stop(), shutting down the app."""
    router = Router()
    router.add_transport(LocalTransport())
    app = AciesApp('test-app', 'test-host', router=router)

    ready = threading.Event()

    @app.on_startup
    def _set_ready(ctx: AciesContext) -> None:
        ready.set()

    @app.thread
    def crasher(ctx: AciesContext, stop: threading.Event) -> None:
        raise RuntimeError('intentional crash')

    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    assert ready.wait(timeout=2.0), 'app did not become ready'
    # do NOT call stop() — the crash must trigger shutdown on its own
    t.join(timeout=3.0)
    assert not t.is_alive(), 'crash did not trigger shutdown'


def test_shutdown_hook_runs_after_thread_joined():
    """Shutdown hook runs after managed threads are joined.

    The thread sets a flag just before returning; the shutdown hook asserts
    the flag is set, proving the thread completed before the hook ran.
    """
    app, ready = _make_app()
    order: list[str] = []

    @app.thread
    def worker(ctx: AciesContext, stop: threading.Event) -> None:
        _ = stop.wait()
        ctx.app.data['thread_done'] = True
        order.append('thread')

    @app.on_shutdown
    def teardown(ctx: AciesContext) -> None:
        assert ctx.app.data.get('thread_done'), 'thread had not finished before shutdown hook'
        order.append('shutdown')

    with running(app, ready=ready):
        pass

    assert order == ['thread', 'shutdown']


def test_multiple_threads_all_stop():
    """All managed threads receive the stop signal and exit."""
    app, ready = _make_app()

    n = 3
    started_events = [threading.Event() for _ in range(n)]
    stopped_flags: list[bool] = [False] * n

    for i in range(n):
        ev = started_events[i]

        @app.thread
        def worker(ctx: AciesContext, stop: threading.Event, ev: threading.Event = ev, i: int = i) -> None:
            ev.set()
            _ = stop.wait()
            stopped_flags[i] = True

    with running(app, ready=ready):
        for ev in started_events:
            assert ev.wait(timeout=2.0), 'a thread never started'

    assert all(stopped_flags), f'not all threads stopped: {stopped_flags}'


def test_thread_per_task_ctx():
    """Each thread gets its own TaskState, not shared with other threads."""
    app, ready = _make_app()

    results: list[str | None] = []
    both_done = threading.Barrier(3)  # 2 threads + test thread

    @app.thread
    def worker_a(ctx: AciesContext, stop: threading.Event) -> None:
        ctx.task.data['id'] = 'a'
        _ = both_done.wait(timeout=2.0)
        results.append(ctx.task.data.get('id'))
        _ = stop.wait()

    @app.thread
    def worker_b(ctx: AciesContext, stop: threading.Event) -> None:
        ctx.task.data['id'] = 'b'
        _ = both_done.wait(timeout=2.0)
        results.append(ctx.task.data.get('id'))
        _ = stop.wait()

    with running(app, ready=ready):
        _ = both_done.wait(timeout=2.0)

    assert sorted(results) == ['a', 'b'], f'task state was shared: {results}'  # pyright: ignore[reportArgumentType]
