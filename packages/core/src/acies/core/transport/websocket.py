"""WebSocketTransport -- browser/UI transport over WebSocket.

Wire protocol (symmetric msgpack [topic: str, payload: bytes]):
  The browser uses clean topic names without the 'ws://' prefix.
  Outbound (server -> browser): strips 'ws://' before sending.
  Inbound  (browser -> server): prepends 'ws://' for internal routing.

Usage::

    app = AciesApp()
    app._router.add_transport(WebSocketTransport(port=8765), prefix='ws://')

    @app.subscribe('ws://control')
    def on_control(ctx, msg: ControlMsg) -> None:
        ...

    @app.schedule(1.0)
    def tick(ctx):
        ctx.publish('ws://dashboard', MyMsg(...))

Browser (inbound)::

    ws.send(msgpack.encode(['control', payload_bytes]));

Browser (outbound)::

    ws.onmessage = async e => {
        const [topic, payload] = decode(new Uint8Array(await e.data.arrayBuffer()));
        // topic is 'dashboard', 'predictions', etc. (no 'ws://' prefix)
    };

query() and advertise() are no-ops -- WebSocket does not support RPC.

Logger: acies.core.transport.websocket
"""

from __future__ import annotations

import logging
import threading

import msgspec
import websockets.sync.server
from websockets.sync.server import ServerConnection

from ._base import MessageHandler

logger = logging.getLogger(__name__)

_PREFIX = 'ws://'


class WsFrame(msgspec.Struct, array_like=True):
    """WebSocket wire frame -- msgpack array [topic, payload].

    array_like=True encodes as a 2-element msgpack array so the browser
    decodes it as: const [topic, payload] = decode(raw).

    To evolve the protocol, add fields with defaults so older clients
    remain compatible:
        timestamp: int = 0
    """

    topic: str
    payload: bytes


class WebSocketTransport:
    """WebSocket server transport for browser and UI clients.

    Each connected client runs in its own thread (websockets sync model).
    All connected clients receive every published message (outbound).
    Inbound frames are decoded as msgpack [topic, payload] and routed via on_message.

    Args:
        host: Interface to bind to. Defaults to '0.0.0.0' (all interfaces).
        port: TCP port to listen on. Defaults to 8765.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 8765) -> None:
        self._host: str = host
        self._port: int = port
        self._on_message: MessageHandler | None = None
        self._clients: set[ServerConnection] = set()
        self._clients_lock: threading.Lock = threading.Lock()
        self._server: websockets.sync.server.WebSocketServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, on_message: MessageHandler) -> None:
        self._on_message = on_message
        self._server = websockets.sync.server.serve(self._handle_client, self._host, self._port)
        self._thread = threading.Thread(target=self._server.serve_forever, name='ws-transport', daemon=True)
        self._thread.start()
        logger.debug('server started on %s:%d', self._host, self._port)

    def stop(self) -> None:
        logger.debug('stop requested')
        if self._server:
            self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.debug('stopped')

    def abort(self) -> None:
        logger.debug('abort requested')
        self.stop()

    def subscribe(self, topic: str) -> None:
        logger.debug('subscribe %r (no-op: WebSocket is a dumb pipe)', topic)

    def unsubscribe(self, topic: str) -> None:
        logger.debug('unsubscribe %r (no-op)', topic)

    def advertise(self, topic: str) -> None:
        logger.debug('advertise %r (no-op: WebSocket does not support RPC)', topic)

    def unadvertise(self, topic: str) -> None:
        logger.debug('unadvertise %r (no-op)', topic)

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        logger.debug(
            'query %r (size=%d bytes, timeout=%d) (no-op: WebSocket does not support RPC)',
            topic,
            len(raw),
            timeout,
        )
        return None

    def publish(self, topic: str, raw: bytes) -> None:
        """Broadcast msgpack [topic, payload] frame to all connected clients.

        Strips the internal 'ws://' prefix so the browser sees clean topics.
        """
        topic = topic.removeprefix(_PREFIX)
        with self._clients_lock:
            clients = list(self._clients)
        if not clients:
            logger.debug('publish %r: no connected clients', topic)
            return
        frame = msgspec.msgpack.encode(WsFrame(topic=topic, payload=raw))
        n_ok, n_err = 0, 0
        for ws in clients:
            try:
                ws.send(frame)
                n_ok += 1
            except Exception:
                logger.exception('send failed to %s', ws.remote_address)
                n_err += 1
        logger.debug('broadcast %r -> %d/%d client(s) (%d bytes)', topic, n_ok, len(clients), len(raw))

    # ---------------------------- per-client thread ---------------------------

    def _handle_client(self, ws: ServerConnection) -> None:
        with self._clients_lock:
            self._clients.add(ws)
        addr = ws.remote_address
        logger.debug('client connected: %s (total: %d)', addr, len(self._clients))
        try:
            for raw_msg in ws:
                if not isinstance(raw_msg, bytes):
                    logger.warning('client %s sent non-binary frame; dropping', addr)
                    continue
                if self._on_message is None:
                    logger.debug('client %s sent frame but on_message not set; dropping', addr)
                    continue
                try:
                    frame = msgspec.msgpack.decode(raw_msg, type=WsFrame)
                    logger.debug('inbound %r from %s (%d bytes)', frame.topic, addr, len(frame.payload))
                    topic = frame.topic if frame.topic.startswith(_PREFIX) else _PREFIX + frame.topic
                    self._on_message(topic, frame.payload, None)
                except (msgspec.DecodeError, ValueError):
                    logger.warning('client %s sent malformed frame (%d bytes); dropping', addr, len(raw_msg))
        except Exception:
            logger.exception('unexpected error in client handler for %s', addr)
        finally:
            with self._clients_lock:
                self._clients.discard(ws)
            logger.debug('client disconnected: %s (total: %d)', addr, len(self._clients))
