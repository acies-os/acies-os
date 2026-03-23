"""Smoke tests for ZenohTransport.

These tests use two in-process ZenohTransport instances in peer mode (no
zenoh router daemon required — UDP multicast discovery handles same-host
communication).

Each test brings up its own pair of transports and tears them down, so the
tests are independent.  They are marked `zenoh` so they can be skipped in
environments without network access:

    uv run pytest -m "not zenoh"  # skip
    uv run pytest -m zenoh        # only zenoh
"""

import threading
import time

import msgspec
import pytest

from acies.corev2.transport import ZenohTransport

# ---------------------------- pub/sub smoke tests ----------------------------


@pytest.mark.zenoh
def test_publish_delivers_to_subscriber():
    """A message published on a topic reaches a subscriber on the same topic."""
    delivered: list[tuple[str, bytes]] = []
    event = threading.Event()

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        delivered.append((topic, raw))
        event.set()

    pub = ZenohTransport()
    sub = ZenohTransport()

    pub.start(lambda t, r, f=None: None)
    sub.start(on_message)
    sub.subscribe('smoke/test/ping')

    # Give zenoh peer discovery a moment to propagate the subscription.
    time.sleep(0.3)

    pub.publish('smoke/test/ping', b'hello')

    assert event.wait(timeout=5.0), 'message not delivered within timeout'
    assert delivered == [('smoke/test/ping', b'hello')]

    pub.stop()
    sub.stop()


@pytest.mark.zenoh
def test_unsubscribed_topic_not_delivered():
    """Messages on an unsubscribed topic are not delivered."""
    delivered: list[str] = []

    pub = ZenohTransport()
    sub = ZenohTransport()

    pub.start(lambda t, r, f=None: None)
    sub.start(lambda t, r, f=None: delivered.append(t))
    sub.subscribe('smoke/test/subscribed')

    time.sleep(0.3)

    pub.publish('smoke/test/other', b'data')

    time.sleep(0.3)
    assert delivered == []

    pub.stop()
    sub.stop()


@pytest.mark.zenoh
def test_wildcard_star_matches_one_segment():
    """Topic pattern 'smoke/*/value' matches 'smoke/unit1/value'."""
    event = threading.Event()

    pub = ZenohTransport()
    sub = ZenohTransport()

    pub.start(lambda t, r, f=None: None)
    sub.start(lambda t, r, f=None: event.set())
    sub.subscribe('smoke/*/value')

    time.sleep(0.3)

    pub.publish('smoke/unit1/value', b'x')

    assert event.wait(timeout=5.0)

    pub.stop()
    sub.stop()


# ------------------------ query/advertise smoke tests ------------------------


class _Pong(msgspec.Struct, frozen=True):
    value: int


@pytest.mark.zenoh
def test_query_returns_reply():
    """query() returns the bytes sent by the advertised handler."""
    server = ZenohTransport()
    client = ZenohTransport()

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        if reply_fn is not None:
            reply_fn(msgspec.msgpack.encode(_Pong(value=42)))

    server.start(on_message)
    server.advertise('smoke/rpc/ping')

    client.start(lambda t, r, f=None: None)

    time.sleep(0.3)

    result = client.query('smoke/rpc/ping', b'', timeout=3.0)

    assert result is not None
    assert msgspec.msgpack.decode(result, type=_Pong) == _Pong(value=42)

    server.stop()
    client.stop()


@pytest.mark.zenoh
def test_query_returns_none_on_timeout():
    """query() returns None when no handler is registered."""
    client = ZenohTransport()
    client.start(lambda t, r, f=None: None)

    result = client.query('smoke/rpc/nonexistent', b'', timeout=0.3)

    assert result is None

    client.stop()
