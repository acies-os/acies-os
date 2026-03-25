"""Integration tests for AciesApp.run() using LocalTransport (no Zenoh needed)."""

import threading
import time
from contextlib import contextmanager

import msgspec
import pytest

from acies.corev2.app import AciesApp
from acies.corev2.context import AciesContext
from acies.corev2.namespace import CtlTopic, Topic
from acies.corev2.router import Router
from acies.corev2.transport import LocalTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app() -> tuple[AciesApp, Router, threading.Event]:
    """Return a new app wired to a LocalTransport, the router, and a ready event.

    The ready event is set by a startup hook — after all specs are subscribed/
    advertised — so callers can wait for the app to be fully wired before
    interacting with it.
    """
    router = Router()
    router.add_transport(LocalTransport())
    app = AciesApp('test-app', 'test-host', router=router)

    ready = threading.Event()

    @app.on_startup
    def _set_ready(ctx: AciesContext) -> None:
        ready.set()

    return app, router, ready


@contextmanager
def running(app: AciesApp, ready: threading.Event | None = None, timeout: float = 2.0):
    """Start app in a background thread; optionally wait for ready; stop and join on exit."""
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    if ready is not None:
        assert ready.wait(timeout=timeout), 'app did not become ready in time'
    try:
        yield t
    finally:
        app.stop()
        t.join(timeout=timeout)
        assert not t.is_alive(), 'app.run() did not return after stop()'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_on_startup_hook_runs():
    app, _, ready = _make_app()
    called: list[bool] = []

    @app.on_startup
    def setup(ctx: AciesContext):
        called.append(True)

    with running(app, ready=ready):
        pass

    assert called == [True]


def test_on_startup_ctx_publish_works():
    """ctx.publish() in a startup hook does not raise and is delivered."""
    app, _, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    delivered = threading.Event()

    @app.subscribe('out/topic')
    def handler(ctx: AciesContext, msg: Msg):
        delivered.set()

    @app.on_startup
    def setup(ctx: AciesContext):
        ctx.publish('out/topic', Msg(value=1))

    with running(app, ready=ready):
        assert delivered.wait(timeout=2.0), 'startup hook publish not delivered'


def test_subscribe_handler_receives_message():
    app, router, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    received: list[int] = []
    done = threading.Event()

    @app.subscribe('sensors/temp')
    def handler(ctx: AciesContext, msg: Msg):
        received.append(msg.value)
        done.set()

    with running(app, ready=ready):
        router.publish('sensors/temp', msgspec.msgpack.encode(Msg(value=42)))
        assert done.wait(timeout=2.0), 'subscriber never fired'

    assert received == [42]


def test_schedule_fires_repeatedly():
    app, _, ready = _make_app()
    count = [0]
    interval = 0.05

    @app.schedule(interval=interval)
    def tick(ctx: AciesContext):
        count[0] += 1

    with running(app, ready=ready):
        time.sleep(interval * 6)

    assert count[0] >= 3, f'expected >=3 fires, got {count[0]}'


def test_service_returns_value():
    app, router, ready = _make_app()

    class Req(msgspec.Struct):
        x: int

    class Resp(msgspec.Struct):
        y: int

    @app.service('rpc/double')
    def double(ctx: AciesContext, msg: Req) -> Resp:
        return Resp(y=msg.x * 2)

    with running(app, ready=ready):
        raw = router.query('rpc/double', msgspec.msgpack.encode(Req(x=21)), timeout=2.0)

    assert raw is not None
    assert msgspec.msgpack.decode(raw, type=Resp).y == 42


def test_ctx_task_state_persists_across_calls():
    """Value written to ctx.task.data in call N is readable in call N+1."""
    app, router, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    second_call_saw: list[int] = []
    second_call = threading.Event()

    @app.subscribe('data/x')
    def handler(ctx: AciesContext, msg: Msg):
        with ctx.task.lock:
            prev = ctx.task.data.get('last')
            ctx.task.data['last'] = msg.value
        if prev is not None:
            second_call_saw.append(prev)
            second_call.set()

    with running(app, ready=ready):
        router.publish('data/x', msgspec.msgpack.encode(Msg(value=10)))
        time.sleep(0.05)
        router.publish('data/x', msgspec.msgpack.encode(Msg(value=20)))
        assert second_call.wait(timeout=2.0), 'second call never fired'

    assert second_call_saw == [10]


def test_ctx_app_state_shared_across_handlers():
    """Value written by one handler is readable by a different handler."""
    app, router, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    read_by_b: list[int] = []
    done = threading.Event()

    @app.subscribe('channel/a')
    def handler_a(ctx: AciesContext, msg: Msg):
        with ctx.app.lock:
            ctx.app.data['shared'] = msg.value

    @app.subscribe('channel/b')
    def handler_b(ctx: AciesContext, msg: Msg):
        with ctx.app.lock:
            val = ctx.app.data.get('shared')
        if val is not None:
            read_by_b.append(val)
            done.set()

    with running(app, ready=ready):
        router.publish('channel/a', msgspec.msgpack.encode(Msg(value=99)))
        time.sleep(0.05)
        router.publish('channel/b', msgspec.msgpack.encode(Msg(value=0)))
        assert done.wait(timeout=2.0), 'handler_b never read shared state'

    assert read_by_b == [99]


def test_subscribe_topic_template_resolved_from_config():
    """A {placeholder} topic is resolved from app.state.config before wiring."""
    app, router, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    received: list[int] = []
    done = threading.Event()

    @app.subscribe('{input_topic}')
    def handler(ctx: AciesContext, msg: Msg):
        received.append(msg.value)
        done.set()

    app.state.config['input_topic'] = 'sensors/temp'

    with running(app, ready=ready):
        router.publish('sensors/temp', msgspec.msgpack.encode(Msg(value=7)))
        assert done.wait(timeout=2.0), 'templated subscriber never fired'

    assert received == [7]


def test_subscribe_topic():
    """Topic resolves to host/name/... at run() time."""
    app, router, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    received: list[int] = []
    done = threading.Event()

    @app.subscribe(Topic('audio/raw'))
    def handler(ctx: AciesContext, msg: Msg):
        received.append(msg.value)
        done.set()

    with running(app, ready=ready):
        router.publish('test-host/test-app/audio/raw', msgspec.msgpack.encode(Msg(value=1)))
        assert done.wait(timeout=2.0), 'Topic subscriber never fired'

    assert received == [1]


def test_service_ctl_topic():
    """CtlTopic resolves to host/name/ctl/... at run() time."""
    app, router, ready = _make_app()

    class Req(msgspec.Struct):
        x: int

    class Resp(msgspec.Struct):
        y: int

    @app.service(CtlTopic('double'))
    def double(ctx: AciesContext, msg: Req) -> Resp:
        return Resp(y=msg.x * 2)

    with running(app, ready=ready):
        raw = router.query(
            'test-host/test-app/ctl/double',
            msgspec.msgpack.encode(Req(x=5)),
            timeout=2.0,
        )

    assert raw is not None
    assert msgspec.msgpack.decode(raw, type=Resp).y == 10


def test_subscribe_format_string():
    """Format string topics resolve from app.state.config at run() time."""
    app, router, ready = _make_app()

    class Msg(msgspec.Struct):
        value: int

    received: list[int] = []
    done = threading.Event()

    @app.subscribe('{input_topic}')
    def handler(ctx: AciesContext, msg: Msg):
        received.append(msg.value)
        done.set()

    app.state.config['input_topic'] = 'sensors/temp'

    with running(app, ready=ready):
        router.publish('sensors/temp', msgspec.msgpack.encode(Msg(value=3)))
        assert done.wait(timeout=2.0), 'format string subscriber never fired'

    assert received == [3]


def test_subscribe_unresolved_template_raises():
    """run() raises ValueError if a topic template key is missing from config."""
    router = Router()
    router.add_transport(LocalTransport())
    app = AciesApp('test', 'host', router=router)

    @app.subscribe('{missing_key}')
    def handler(ctx: AciesContext, msg: object) -> None:
        pass

    with pytest.raises(ValueError, match='missing_key'):
        app.run()


def test_stop_unblocks_run_and_shutdown_hook_runs():
    app, _, ready = _make_app()
    shutdown_called: list[bool] = []

    @app.on_shutdown
    def teardown(ctx: AciesContext):
        shutdown_called.append(True)

    with running(app, ready=ready):
        pass  # stop() is called by the context manager

    assert shutdown_called == [True]
