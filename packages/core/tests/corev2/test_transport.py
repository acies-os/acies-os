import threading
import time

import msgspec

from acies.corev2.transport import LocalTransport, _topic_matches

# ---------------------------------------------------------------------------
# _topic_matches unit tests
# ---------------------------------------------------------------------------


def test_exact_match():
    assert _topic_matches('sensors/temp', 'sensors/temp')
    assert not _topic_matches('sensors/temp', 'sensors/other')


def test_single_wildcard_matches_one_chunk():
    assert _topic_matches('sensors/*/geo', 'sensors/unit1/geo')
    assert _topic_matches('sensors/*/geo', 'sensors/abc/geo')


def test_single_wildcard_does_not_match_wrong_depth():
    # * must match exactly one chunk — sensors/geo has no middle chunk
    assert not _topic_matches('sensors/*/geo', 'sensors/geo')
    # * does not span slashes
    assert not _topic_matches('sensors/*/geo', 'sensors/unit1/unit2/geo')


def test_double_star_wildcard_matches_multiple_segments():
    assert _topic_matches('sensors/**', 'sensors/mic')
    assert _topic_matches('sensors/**', 'sensors/unit1/geo')
    assert _topic_matches('sensors/**', 'sensors/a/b/c')


# ---------------------------------------------------------------------------
# LocalTransport publish / subscribe tests
# ---------------------------------------------------------------------------


def test_publish_delivers_to_subscriber():
    transport = LocalTransport()
    delivered: list[tuple[str, bytes]] = []
    event = threading.Event()

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        delivered.append((topic, raw))
        event.set()

    transport.start(on_message)
    transport.subscribe('sensors/temp')
    transport.publish('sensors/temp', b'raw_data')

    assert event.wait(timeout=2.0), 'message not delivered within timeout'
    assert delivered == [('sensors/temp', b'raw_data')]
    transport.stop()


def test_unsubscribed_topic_not_delivered():
    transport = LocalTransport()
    delivered: list[str] = []

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        delivered.append(topic)

    transport.start(on_message)
    transport.subscribe('sensors/temp')
    transport.publish('sensors/other', b'data')

    time.sleep(0.05)  # give the receiver thread time to process
    assert delivered == []
    transport.stop()


def test_wildcard_star_matches_one_chunk():
    transport = LocalTransport()
    event = threading.Event()

    transport.start(lambda t, r, f=None: event.set())
    transport.subscribe('sensors/*/geo')
    transport.publish('sensors/unit1/geo', b'data')

    assert event.wait(timeout=2.0)
    transport.stop()


def test_wildcard_star_no_match_wrong_depth():
    transport = LocalTransport()
    delivered: list[str] = []

    transport.start(lambda t, r, f=None: delivered.append(t))
    transport.subscribe('sensors/*/geo')
    transport.publish('sensors/geo', b'data')

    time.sleep(0.05)
    assert delivered == []
    transport.stop()


# ---------------------------------------------------------------------------
# LocalTransport query / advertise tests
# ---------------------------------------------------------------------------


class _Reply(msgspec.Struct, frozen=True):
    value: int


def test_query_returns_reply():
    transport = LocalTransport()

    def on_message(topic: str, raw: bytes, send_bytes=None) -> None:
        # Simulate dispatch: decode request, compute reply, encode and send.
        if send_bytes is not None:
            threading.Thread(
                target=send_bytes,
                args=(msgspec.msgpack.encode(_Reply(value=42)),),
                daemon=True,
            ).start()

    transport.start(on_message)
    transport.advertise('rpc/test')

    result = transport.query('rpc/test', b'query_raw', timeout=2.0)

    assert result == msgspec.msgpack.encode(_Reply(value=42))
    transport.stop()


def test_query_returns_none_on_timeout():
    transport = LocalTransport()
    transport.start(lambda t, r, sb=None: None)
    # No advertiser registered — query should time out and return None.

    result = transport.query('rpc/nonexistent', b'query', timeout=0.05)

    assert result is None
    transport.stop()


def test_query_with_wildcard_advertiser():
    """Advertise on a wildcard pattern; query on a matching concrete topic."""
    transport = LocalTransport()

    def on_message(topic: str, raw: bytes, send_bytes=None) -> None:
        if send_bytes is not None:
            threading.Thread(
                target=send_bytes,
                args=(msgspec.msgpack.encode(_Reply(value=7)),),
                daemon=True,
            ).start()

    transport.start(on_message)
    transport.advertise('rpc/*/status')

    result = transport.query('rpc/node1/status', b'', timeout=2.0)

    assert result == msgspec.msgpack.encode(_Reply(value=7))
    transport.stop()
