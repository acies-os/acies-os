import threading
import time

import msgspec
import pytest

from acies.corev2.executor import Executor
from acies.corev2.router import Router
from acies.corev2.task import Job, ServiceSpec, SubscriberSpec
from acies.corev2.transport import LocalTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Ping(msgspec.Struct, frozen=True):
    value: int = 0


class _Pong(msgspec.Struct, frozen=True):
    value: int = 0


def _make_subscriber(topic: str, msg_type: type | None = None) -> SubscriberSpec:
    def handler(ctx, msg):
        pass

    return SubscriberSpec(name='sub', fn=handler, topics=(topic,), msg_type=msg_type)


def _make_service(topic: str, msg_type: type | None = None) -> ServiceSpec:
    def handler(ctx, msg):
        return _Pong(value=msg.value * 2)

    return ServiceSpec(name='svc', fn=handler, topics=(topic,), msg_type=msg_type)


def _stop(router: Router, executor: Executor) -> None:
    router.stop()
    executor.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_transport_raises():
    router = Router()
    with pytest.raises(RuntimeError):
        router.publish('some/topic', b'bytes')


def test_prefix_routing_selects_correct_transport():
    """ws:// prefix transport is used for ws:// topics; default handles everything else."""
    default_transport = LocalTransport()
    ws_transport = LocalTransport()

    router = Router()
    router.add_transport(default_transport)
    router.add_transport(ws_transport, prefix='ws://')

    received_default: list[bytes] = []
    received_ws: list[bytes] = []

    original_default_pub = default_transport.publish
    original_ws_pub = ws_transport.publish

    def patched_default(topic, raw):
        received_default.append(raw)
        original_default_pub(topic, raw)

    def patched_ws(topic, raw):
        received_ws.append(raw)
        original_ws_pub(topic, raw)

    default_transport.publish = patched_default
    ws_transport.publish = patched_ws

    ex = Executor()

    def dispatch(job: Job):
        pass

    ex.start(dispatch)

    spec_default = _make_subscriber('sensors/temp')
    spec_ws = _make_subscriber('ws://ui/data')

    router.start(ex)
    router.subscribe('sensors/temp', spec_default)
    router.subscribe('ws://ui/data', spec_ws)

    router.publish('sensors/temp', msgspec.msgpack.encode(_Ping(value=1)))
    router.publish('ws://ui/data', msgspec.msgpack.encode(_Ping(value=2)))

    time.sleep(0.1)
    _stop(router, ex)

    assert len(received_default) == 1
    assert len(received_ws) == 1


def test_subscribe_and_publish_delivers_job():
    """A published message reaches the matching subscriber via the executor."""
    router = Router()
    transport = LocalTransport()
    router.add_transport(transport)

    received: list[_Ping] = []
    done = threading.Event()

    spec = _make_subscriber('sensors/temp', msg_type=_Ping)

    ex = Executor()

    def dispatch(job: Job):
        msg = msgspec.msgpack.decode(job.raw, type=_Ping)
        received.append(msg)
        done.set()

    ex.start(dispatch)
    router.start(ex)
    router.subscribe('sensors/temp', spec)

    router.publish('sensors/temp', msgspec.msgpack.encode(_Ping(value=7)))

    assert done.wait(timeout=2.0), 'handler never fired'
    _stop(router, ex)

    assert len(received) == 1
    assert received[0].value == 7


def test_multiple_subscribers_same_topic():
    """All specs subscribed to the same topic each receive a Job."""
    router = Router()
    transport = LocalTransport()
    router.add_transport(transport)

    def make_spec(name: str) -> SubscriberSpec:
        def handler(ctx, msg):
            pass

        return SubscriberSpec(name=name, fn=handler, topics=('data/x',), msg_type=_Ping)

    spec_a = make_spec('a')
    spec_b = make_spec('b')

    done_a = threading.Event()
    done_b = threading.Event()

    ex = Executor()

    def dispatch(job: Job):
        if job.spec is spec_a:
            done_a.set()
        elif job.spec is spec_b:
            done_b.set()

    ex.start(dispatch)
    router.start(ex)

    router.subscribe('data/x', spec_a)
    router.subscribe('data/x', spec_b)

    router.publish('data/x', msgspec.msgpack.encode(_Ping(value=1)))

    assert done_a.wait(timeout=2.0), 'spec_a never received job'
    assert done_b.wait(timeout=2.0), 'spec_b never received job'

    _stop(router, ex)


def test_service_job_reply_fn_end_to_end():
    """query() returns the handler's return value after encode/decode round-trip."""
    router = Router()
    transport = LocalTransport()
    router.add_transport(transport)

    spec = _make_service('rpc/ping', msg_type=_Ping)
    handler_done = threading.Event()

    ex = Executor()

    def dispatch(job: Job):
        msg = msgspec.msgpack.decode(job.raw, type=_Ping)
        result = _Pong(value=msg.value * 2)
        handler_done.set()
        if job.reply_fn is not None:
            job.reply_fn(msgspec.msgpack.encode(result))

    ex.start(dispatch)
    router.start(ex)
    router.advertise('rpc/ping', spec)

    raw_reply = router.query('rpc/ping', msgspec.msgpack.encode(_Ping(value=21)), timeout=2.0)

    assert handler_done.is_set()
    assert raw_reply is not None
    reply = msgspec.msgpack.decode(raw_reply, type=_Pong)
    assert reply.value == 42

    _stop(router, ex)


def test_wildcard_subscription_receives_matching_messages():
    """A wildcard subscriber pattern receives all matching published topics."""
    router = Router()
    transport = LocalTransport()
    router.add_transport(transport)

    received_count = [0]
    done = threading.Event()

    spec = _make_subscriber('sensors/*/temp', msg_type=_Ping)

    ex = Executor()

    def dispatch(job: Job):
        received_count[0] += 1
        if received_count[0] == 2:
            done.set()

    ex.start(dispatch)
    router.start(ex)
    router.subscribe('sensors/*/temp', spec)

    router.publish('sensors/unit1/temp', msgspec.msgpack.encode(_Ping(value=1)))
    router.publish('sensors/unit2/temp', msgspec.msgpack.encode(_Ping(value=2)))

    assert done.wait(timeout=2.0), 'wildcard subscriber did not receive both messages'
    _stop(router, ex)


def test_unsubscribed_topic_not_delivered():
    """Messages on topics with no matching subscriber are silently dropped."""
    router = Router()
    transport = LocalTransport()
    router.add_transport(transport)

    received: list[_Ping] = []
    spec = _make_subscriber('sensors/temp', msg_type=_Ping)

    ex = Executor()

    def dispatch(job: Job):
        received.append(msgspec.msgpack.decode(job.raw, type=_Ping))

    ex.start(dispatch)
    router.start(ex)
    router.subscribe('sensors/temp', spec)

    router.publish('sensors/other', msgspec.msgpack.encode(_Ping(value=99)))
    router.publish('sensors/temp', msgspec.msgpack.encode(_Ping(value=1)))

    time.sleep(0.1)
    _stop(router, ex)

    assert len(received) == 1
    assert received[0].value == 1
