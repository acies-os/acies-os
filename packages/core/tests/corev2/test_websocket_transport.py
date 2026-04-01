import threading
import time

import msgspec
import websockets.sync.client

from acies.corev2.transport.websocket import WebSocketTransport, WsFrame

# ---------------------------------- helpers ----------------------------------

_HOST = '127.0.0.1'
_PORT = 19765  # avoid collision with any real server


def _connect() -> websockets.sync.client.ClientConnection:
    return websockets.sync.client.connect(f'ws://{_HOST}:{_PORT}')


def _encode(topic: str, payload: bytes) -> bytes:
    return msgspec.msgpack.encode(WsFrame(topic=topic, payload=payload))


def _decode(raw: bytes) -> WsFrame:
    return msgspec.msgpack.decode(raw, type=WsFrame)


# ---------------------------------- lifecyle ----------------------------------


def test_start_stop():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    ws = _connect()
    ws.close()
    t.stop()


def test_multiple_clients_connect():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    clients = [_connect() for _ in range(3)]
    time.sleep(0.05)
    assert len(t._clients) == 3
    for ws in clients:
        ws.close()
    t.stop()


def test_client_removed_on_disconnect():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    ws = _connect()
    time.sleep(0.05)
    assert len(t._clients) == 1
    ws.close()
    time.sleep(0.05)
    assert len(t._clients) == 0
    t.stop()


# ---------------------------------- outbound ----------------------------------


def test_publish_delivers_to_client():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    ws = _connect()
    time.sleep(0.05)

    t.publish('ws://dashboard', b'hello')

    raw = ws.recv(timeout=2.0)
    frame = _decode(raw)
    assert frame.topic == 'ws://dashboard'
    assert frame.payload == b'hello'

    ws.close()
    t.stop()


def test_publish_delivers_to_all_clients():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    clients = [_connect() for _ in range(3)]
    time.sleep(0.05)

    t.publish('ws://dashboard', b'broadcast')

    for ws in clients:
        frame = _decode(ws.recv(timeout=2.0))
        assert frame.topic == 'ws://dashboard'
        assert frame.payload == b'broadcast'
        ws.close()

    t.stop()


def test_publish_no_clients_does_not_raise():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    t.publish('ws://dashboard', b'nobody home')  # should not raise
    t.stop()


def test_publish_preserves_topic_in_frame():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    ws = _connect()
    time.sleep(0.05)

    t.publish('ws://inference', b'\x00\x01\x02')

    frame = _decode(ws.recv(timeout=2.0))
    assert frame.topic == 'ws://inference'

    ws.close()
    t.stop()


# ---------------------------------- inbound ----------------------------------


def test_inbound_frame_delivered_to_on_message():
    received: list[tuple[str, bytes]] = []
    event = threading.Event()

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        received.append((topic, raw))
        event.set()

    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(on_message)
    ws = _connect()
    time.sleep(0.05)

    ws.send(_encode('ws://control', b'payload'))

    assert event.wait(timeout=2.0), 'on_message not called'
    assert received == [('ws://control', b'payload')]

    ws.close()
    t.stop()


def test_inbound_topic_routed_correctly():
    topics: list[str] = []
    event = threading.Event()

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        topics.append(topic)
        event.set()

    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(on_message)
    ws = _connect()
    time.sleep(0.05)

    ws.send(_encode('ws://config', b'data'))

    assert event.wait(timeout=2.0)
    assert topics == ['ws://config']

    ws.close()
    t.stop()


def test_inbound_malformed_frame_dropped():
    received: list[bytes] = []

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        received.append(raw)

    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(on_message)
    ws = _connect()
    time.sleep(0.05)

    ws.send(b'\xff\xfe invalid msgpack')  # malformed

    time.sleep(0.05)
    assert received == []  # nothing delivered

    ws.close()
    t.stop()


def test_inbound_text_frame_dropped():
    received: list[bytes] = []

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        received.append(raw)

    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(on_message)
    ws = _connect()
    time.sleep(0.05)

    ws.send('this is a text frame')  # should be binary

    time.sleep(0.05)
    assert received == []

    ws.close()
    t.stop()


def test_multiple_inbound_frames_from_same_client():
    received: list[tuple[str, bytes]] = []
    done = threading.Event()

    def on_message(topic: str, raw: bytes, reply_fn=None) -> None:
        received.append((topic, raw))
        if len(received) == 3:
            done.set()

    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(on_message)
    ws = _connect()
    time.sleep(0.05)

    for i in range(3):
        ws.send(_encode(f'ws://ch{i}', f'payload{i}'.encode()))

    assert done.wait(timeout=2.0)
    assert [(t, p) for t, p in received] == [
        ('ws://ch0', b'payload0'),
        ('ws://ch1', b'payload1'),
        ('ws://ch2', b'payload2'),
    ]

    ws.close()
    t.stop()


# ----------------------------------- no-op -----------------------------------


def test_no_ops_do_not_raise():
    t = WebSocketTransport(host=_HOST, port=_PORT)
    t.start(lambda topic, raw, reply_fn: None)
    t.subscribe('ws://x')
    t.unsubscribe('ws://x')
    t.advertise('ws://x')
    t.unadvertise('ws://x')
    assert t.query('ws://x', b'', 0.1) is None
    t.stop()
