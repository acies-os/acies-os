"""Tests for default control handlers: KV path helpers and the ctl/kv service."""

import threading
from contextlib import contextmanager

import msgspec
import pytest

from acies.corev2._control import _del_path, _get_path, _set_path  # pyright: ignore[reportPrivateUsage]
from acies.corev2.app import AciesApp
from acies.corev2.context import AciesContext
from acies.corev2.msg import (
    AciesDel,
    AciesGet,
    AciesKvRequest,
    AciesKvResponse,
    AciesSet,
    Err,
    Ok,
)
from acies.corev2.router import Router
from acies.corev2.transport import LocalTransport

# ---------------------------------- helpers ----------------------------------


def _make_app() -> tuple[AciesApp, Router, threading.Event]:
    router = Router()
    router.add_transport(LocalTransport())
    app = AciesApp('test-app', 'test-host', router=router)
    ready = threading.Event()

    @app.on_startup
    def _set_ready(ctx: AciesContext) -> None:
        ready.set()

    return app, router, ready


@contextmanager
def running(app: AciesApp, ready: threading.Event, timeout: float = 2.0):
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    assert ready.wait(timeout=timeout), 'app did not become ready'
    try:
        yield
    finally:
        app.stop()
        t.join(timeout=timeout)


def _kv_query(
    router: Router,
    ops: list,
    source: str = 'test',
    timeout: float = 2.0,
) -> AciesKvResponse:
    raw = router.query(
        'test-host/test-app/ctl/kv',
        msgspec.msgpack.encode(AciesKvRequest(source=source, timestamp=0, ops=ops)),
        timeout=timeout,
    )
    assert raw is not None, 'kv query timed out'
    return msgspec.msgpack.decode(raw, type=AciesKvResponse)


# --------------------------- tests for path helpers ---------------------------


class TestGetPath:
    def test_top_level(self):
        assert _get_path({'a': 1}, ['a']) == 1

    def test_nested(self):
        assert _get_path({'a': {'b': {'c': 42}}}, ['a', 'b', 'c']) == 42

    def test_returns_subtree(self):
        assert _get_path({'a': {'b': 1}}, ['a']) == {'b': 1}

    def test_missing_top_level(self):
        with pytest.raises(KeyError) as exc:
            _get_path({}, ['missing'])
        assert exc.value.args[0] == 'missing'

    def test_missing_nested(self):
        with pytest.raises(KeyError) as exc:
            _get_path({'a': {'b': 1}}, ['a', 'x'])
        assert exc.value.args[0] == 'x'

    def test_through_non_dict(self):
        with pytest.raises(KeyError) as exc:
            _get_path({'a': 'scalar'}, ['a', 'b'])
        assert exc.value.args[0] == 'b'


class TestSetPath:
    def test_top_level(self):
        config = {'a': 1}
        _set_path(config, ['a'], 99)
        assert config['a'] == 99

    def test_nested(self):
        config = {'a': {'b': 1}}
        _set_path(config, ['a', 'b'], 42)
        assert config['a']['b'] == 42

    def test_missing_top_level(self):
        with pytest.raises(KeyError) as exc:
            _set_path({}, ['missing'], 1)
        assert exc.value.args[0] == 'missing'

    def test_missing_nested(self):
        with pytest.raises(KeyError) as exc:
            _set_path({'a': {'b': 1}}, ['a', 'x'], 1)
        assert exc.value.args[0] == 'x'

    def test_missing_intermediate(self):
        with pytest.raises(KeyError) as exc:
            _set_path({}, ['missing', 'b'], 1)
        assert exc.value.args[0] == 'missing'

    def test_intermediate_is_scalar(self):
        with pytest.raises(KeyError) as exc:
            _set_path({'a': 'scalar'}, ['a', 'b'], 1)
        assert exc.value.args[0] == 'a'


class TestDelPath:
    def test_top_level(self):
        config = {'a': 1, 'b': 2}
        _del_path(config, ['a'])
        assert 'a' not in config

    def test_nested(self):
        config = {'a': {'b': 1, 'c': 2}}
        _del_path(config, ['a', 'b'])
        assert config == {'a': {'c': 2}}

    def test_missing_key(self):
        with pytest.raises(KeyError):
            _del_path({'a': 1}, ['missing'])

    def test_missing_nested(self):
        with pytest.raises(KeyError):
            _del_path({'a': {'b': 1}}, ['a', 'missing'])


# -------------------------- tests for ctl/kv service --------------------------


class TestKvGet:
    def test_existing_key(self):
        app, router, ready = _make_app()
        app.state.config['threshold'] = 0.5
        with running(app, ready):
            resp = _kv_query(router, [AciesGet(key=['threshold'])])
        assert resp.results == [Ok(value=0.5)]

    def test_nested_key(self):
        app, router, ready = _make_app()
        app.state.config['model'] = {'threshold': 0.8, 'window': 100}
        with running(app, ready):
            resp = _kv_query(router, [AciesGet(key=['model', 'threshold'])])
        assert resp.results == [Ok(value=0.8)]

    def test_subtree(self):
        app, router, ready = _make_app()
        app.state.config['model'] = {'threshold': 0.8}
        with running(app, ready):
            resp = _kv_query(router, [AciesGet(key=['model'])])
        assert resp.results == [Ok(value={'threshold': 0.8})]

    def test_missing_key(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesGet(key=['missing'])])
        assert isinstance(resp.results[0], Err)
        assert 'missing' in resp.results[0].reason

    def test_empty_path(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesGet(key=[])])
        assert isinstance(resp.results[0], Err)
        assert 'empty' in resp.results[0].reason

    def test_sys_key_readable(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesGet(key=['sys', 'host'])])
        assert resp.results == [Ok(value='test-host')]


class TestKvSet:
    def test_existing_key(self):
        app, router, ready = _make_app()
        app.state.config['threshold'] = 0.5
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=['threshold'], value=0.9)])
            assert resp.results == [Ok()]
        assert app.state.config['threshold'] == 0.9

    def test_nested_key(self):
        app, router, ready = _make_app()
        app.state.config['model'] = {'threshold': 0.5}
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=['model', 'threshold'], value=0.9)])
            assert resp.results == [Ok()]
        assert app.state.config['model']['threshold'] == 0.9

    def test_missing_key(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=['missing'], value=1)])
        assert isinstance(resp.results[0], Err)
        assert 'missing' in resp.results[0].reason

    def test_sys_state_allowed(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=['sys', 'state'], value='paused')])
            assert resp.results == [Ok()]
        assert app.state.config['sys']['state'] == 'paused'

    def test_sys_host_protected(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=['sys', 'host'], value='other')])
        assert isinstance(resp.results[0], Err)
        assert resp.results[0].reason == 'key_protected'

    def test_sys_subtree_protected(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=['sys'], value={})])
        assert isinstance(resp.results[0], Err)
        assert resp.results[0].reason == 'key_protected'

    def test_empty_path(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesSet(key=[], value=1)])
        assert isinstance(resp.results[0], Err)
        assert 'empty' in resp.results[0].reason


class TestKvDel:
    def test_existing_key(self):
        app, router, ready = _make_app()
        app.state.config['temp'] = 42
        with running(app, ready):
            resp = _kv_query(router, [AciesDel(key=['temp'])])
            assert resp.results == [Ok()]
        assert 'temp' not in app.state.config

    def test_nested_key(self):
        app, router, ready = _make_app()
        app.state.config['model'] = {'threshold': 0.5, 'window': 100}
        with running(app, ready):
            resp = _kv_query(router, [AciesDel(key=['model', 'threshold'])])
            assert resp.results == [Ok()]
        assert app.state.config['model'] == {'window': 100}

    def test_missing_key(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesDel(key=['missing'])])
        assert isinstance(resp.results[0], Err)
        assert 'missing' in resp.results[0].reason

    def test_sys_protected(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesDel(key=['sys', 'state'])])
        assert isinstance(resp.results[0], Err)
        assert resp.results[0].reason == 'key_protected'

    def test_empty_path(self):
        app, router, ready = _make_app()
        with running(app, ready):
            resp = _kv_query(router, [AciesDel(key=[])])
        assert isinstance(resp.results[0], Err)
        assert 'empty' in resp.results[0].reason


class TestKvMixed:
    def test_multiple_ops_aligned(self):
        """Results are positionally aligned with ops."""
        app, router, ready = _make_app()
        app.state.config['a'] = 1
        app.state.config['b'] = 2
        with running(app, ready):
            resp = _kv_query(
                router,
                [
                    AciesGet(key=['a']),
                    AciesSet(key=['b'], value=99),
                    AciesGet(key=['missing']),
                    AciesGet(key=['b']),
                ],
            )
        assert len(resp.results) == 4
        assert resp.results[0] == Ok(value=1)
        assert resp.results[1] == Ok()
        assert isinstance(resp.results[2], Err)
        assert resp.results[3] == Ok(value=99)

    def test_failed_op_does_not_abort_batch(self):
        """A failed op in the middle does not prevent subsequent ops."""
        app, router, ready = _make_app()
        app.state.config['x'] = 10
        with running(app, ready):
            resp = _kv_query(
                router,
                [
                    AciesGet(key=['missing']),
                    AciesGet(key=['x']),
                ],
            )
        assert isinstance(resp.results[0], Err)
        assert resp.results[1] == Ok(value=10)
