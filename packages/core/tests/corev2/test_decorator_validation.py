"""Tests for decorator parameter validation across all AciesApp decorators.

Covers:
  - Missing required parameters raise TypeError at decoration time
  - **kwargs functions bypass name checks (all keyword args are absorbed)
  - Valid signatures (including non-standard param order) are accepted
"""

import threading

import msgspec
import pytest

from acies.corev2.app import AciesApp
from acies.corev2.context import AciesContext
from acies.corev2.router import Router
from acies.corev2.transport import LocalTransport


def _make_app() -> AciesApp:
    router = Router()
    router.add_transport(LocalTransport())
    return AciesApp('test', 'host', router=router)


# --------------------------------- on_startup ---------------------------------


def test_on_startup_missing_ctx_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='on_startup.*ctx'):

        @app.on_startup
        def setup() -> None:
            pass


def test_on_startup_valid():
    app = _make_app()

    @app.on_startup
    def setup(ctx: AciesContext) -> None:
        pass


def test_on_startup_kwargs_accepted():
    app = _make_app()

    @app.on_startup
    def setup(**kwargs) -> None:
        pass


# -------------------------------- on_shutdown --------------------------------


def test_on_shutdown_missing_ctx_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='on_shutdown.*ctx'):

        @app.on_shutdown
        def teardown() -> None:
            pass


def test_on_shutdown_valid():
    app = _make_app()

    @app.on_shutdown
    def teardown(ctx: AciesContext) -> None:
        pass


def test_on_shutdown_kwargs_accepted():
    app = _make_app()

    @app.on_shutdown
    def teardown(**kwargs) -> None:
        pass


# ---------------------------------- schedule ----------------------------------


def test_schedule_missing_ctx_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='schedule.*ctx'):

        @app.schedule(interval=1.0)
        def on_tick() -> None:
            pass


def test_schedule_valid():
    app = _make_app()

    @app.schedule(interval=1.0)
    def on_tick(ctx: AciesContext) -> None:
        pass


def test_schedule_kwargs_accepted():
    app = _make_app()

    @app.schedule(interval=1.0)
    def on_tick(**kwargs) -> None:
        pass


# --------------------------------- subscribe ---------------------------------


def test_subscribe_missing_ctx_raises():
    app = _make_app()

    class Msg(msgspec.Struct):
        value: int

    with pytest.raises(TypeError, match='subscriber.*ctx'):

        @app.subscribe('topic')
        def handler(msg: Msg) -> None:
            pass


def test_subscribe_missing_msg_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='subscriber.*msg'):

        @app.subscribe('topic')
        def handler(ctx: AciesContext) -> None:
            pass


def test_subscribe_valid():
    app = _make_app()

    class Msg(msgspec.Struct):
        value: int

    @app.subscribe('topic')
    def handler(ctx: AciesContext, msg: Msg) -> None:
        pass


def test_subscribe_kwargs_accepted():
    app = _make_app()

    @app.subscribe('topic')
    def handler(**kwargs) -> None:
        pass


def test_subscribe_param_order_flexible():
    """Param order does not matter since handlers are called with keyword args."""
    app = _make_app()

    class Msg(msgspec.Struct):
        value: int

    @app.subscribe('topic')
    def handler(msg: Msg, ctx: AciesContext) -> None:
        pass


# ---------------------------------- service ----------------------------------


class _Req(msgspec.Struct):
    x: int


class _Resp(msgspec.Struct):
    y: int


def test_service_missing_ctx_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='service.*ctx'):

        @app.service('topic')
        def handler(msg: _Req) -> _Resp:
            return _Resp(y=0)


def test_service_missing_msg_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='service.*msg'):

        @app.service('topic')
        def handler(ctx: AciesContext) -> _Resp:
            return _Resp(y=0)


def test_service_missing_return_type_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='return type'):

        @app.service('topic')
        def handler(ctx: AciesContext, msg: _Req):
            pass


def test_service_bare_struct_msg_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='specific type annotation'):

        @app.service('topic')
        def handler(ctx: AciesContext, msg: msgspec.Struct) -> _Resp:
            return _Resp(y=0)


def test_service_valid():
    app = _make_app()

    @app.service('topic')
    def handler(ctx: AciesContext, msg: _Req) -> _Resp:
        return _Resp(y=msg.x)


def test_service_kwargs_accepted():
    app = _make_app()

    @app.service('topic')
    def handler(**kwargs) -> _Resp:
        return _Resp(y=0)


def test_service_param_order_flexible():
    app = _make_app()

    @app.service('topic')
    def handler(msg: _Req, ctx: AciesContext) -> _Resp:
        return _Resp(y=msg.x)


# ----------------------------------- thread -----------------------------------


def test_thread_missing_ctx_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='thread.*ctx'):

        @app.thread
        def worker(stop: threading.Event) -> None:
            pass


def test_thread_missing_stop_raises():
    app = _make_app()
    with pytest.raises(TypeError, match='thread.*stop'):

        @app.thread
        def worker(ctx: AciesContext) -> None:
            pass


def test_thread_valid():
    app = _make_app()

    @app.thread
    def worker(ctx: AciesContext, stop: threading.Event) -> None:
        pass


def test_thread_kwargs_accepted():
    app = _make_app()

    @app.thread
    def worker(**kwargs) -> None:
        pass


def test_thread_param_order_flexible():
    app = _make_app()

    @app.thread
    def worker(stop: threading.Event, ctx: AciesContext) -> None:
        pass
