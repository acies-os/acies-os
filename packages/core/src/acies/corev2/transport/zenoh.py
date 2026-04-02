"""ZenohTransport -- the primary production backend.

Each AciesApp holds one ZenohTransport. Cross-node and cross-process
routing is handled transparently by the zenoh network. Same-host IPC
uses UDP multicast discovery (no daemon required); for explicit
endpoints pass connect/listen endpoint lists to the constructor.

Logger: acies.corev2.transport.zenoh
"""

from __future__ import annotations

import json
import logging
import threading

import zenoh

from ._base import MessageHandler

logger = logging.getLogger(__name__)


class ZenohTransport:
    """Zenoh-backed transport.

    Args:
        mode:    Zenoh session mode: 'client' or 'peer'.
        connect: Endpoints to connect to (e.g. ['tcp/router:7447']).
        listen:  Endpoints to listen on in peer mode.
    """

    def __init__(
        self,
        mode: str = 'client',
        connect: list[str] | None = None,
        listen: list[str] | None = None,
    ) -> None:
        cfg: zenoh.Config = zenoh.Config()
        cfg.insert_json5('mode', f'"{mode}"')  # pyright: ignore[reportUnknownMemberType]
        if connect:
            cfg.insert_json5('connect/endpoints', json.dumps(connect))  # pyright: ignore[reportUnknownMemberType]
        if listen:
            cfg.insert_json5('listen/endpoints', json.dumps(listen))  # pyright: ignore[reportUnknownMemberType]
        self._config: zenoh.Config = cfg
        self._session: zenoh.Session | None = None
        self._on_message: MessageHandler | None = None
        self._subscribers: dict[str, zenoh.Subscriber[None]] = {}
        self._queryables: dict[str, zenoh.Queryable[None]] = {}

    def start(self, on_message: MessageHandler) -> None:
        """Open the zenoh session and store the inbound message callback."""
        self._on_message = on_message
        self._session = zenoh.open(self._config)
        logger.debug('session opened')

    def stop(self) -> None:
        """Undeclare all subscribers/queryables and close the session."""
        n_subs = len(self._subscribers)
        n_qbs = len(self._queryables)
        for sub in self._subscribers.values():
            sub.undeclare()  # pyright: ignore[reportUnknownMemberType]
        self._subscribers.clear()
        for qb in self._queryables.values():
            qb.undeclare()  # pyright: ignore[reportUnknownMemberType]
        self._queryables.clear()
        if self._session is not None:
            self._session.close()  # pyright: ignore[reportUnknownMemberType]
            self._session = None
        logger.debug('session closed (%d subscribers, %d queryables undeclared)', n_subs, n_qbs)

    def abort(self) -> None:
        """Immediate shutdown -- same as stop() for zenoh."""
        logger.debug('abort requested')
        self.stop()

    def subscribe(self, topic: str) -> None:
        """Declare a zenoh subscriber that forwards samples to on_message."""
        assert self._session is not None, 'call start() before subscribe()'

        def _on_sample(sample: zenoh.Sample) -> None:
            assert self._on_message is not None, 'on_message callback must be set before subscribing'
            self._on_message(str(sample.key_expr), bytes(sample.payload), None)

        self._subscribers[topic] = self._session.declare_subscriber(topic, _on_sample)
        logger.debug('subscribed %r', topic)

    def unsubscribe(self, topic: str) -> None:
        sub = self._subscribers.pop(topic, None)
        if sub is not None:
            sub.undeclare()  # pyright: ignore[reportUnknownMemberType]
            logger.debug('unsubscribed %r', topic)

    def publish(self, topic: str, raw: bytes) -> None:
        """Put raw bytes to topic."""
        assert self._session is not None, 'call start() before publish()'
        self._session.put(topic, raw)  # pyright: ignore[reportUnknownMemberType]
        logger.debug('publish %r (%d bytes)', topic, len(raw))

    def advertise(self, topic: str) -> None:
        """Declare a zenoh queryable.

        Creates a reply_fn closure over query.reply() and passes it as the
        third argument to on_message so the dispatch layer can call it after
        the handler returns.
        """
        assert self._session is not None, 'call start() before advertise()'

        def _on_query(query: zenoh.Query) -> None:
            if self._on_message is None:
                return
            raw = bytes(query.payload) if query.payload is not None else b''

            def reply_fn(encoded: bytes) -> None:
                query.reply(query.key_expr, encoded)  # pyright: ignore[reportUnknownMemberType]

            self._on_message(str(query.key_expr), raw, reply_fn)

        self._queryables[topic] = self._session.declare_queryable(topic, _on_query)
        logger.debug('advertised %r', topic)

    def unadvertise(self, topic: str) -> None:
        qb = self._queryables.pop(topic, None)
        if qb is not None:
            qb.undeclare()  # pyright: ignore[reportUnknownMemberType]
            logger.debug('unadvertised %r', topic)

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Send a zenoh get and block until a reply arrives or timeout elapses."""
        assert self._session is not None, 'call start() before query()'

        event = threading.Event()
        result: list[bytes | None] = [None]

        def _on_reply(reply: zenoh.Reply) -> None:
            sample = reply.ok
            if sample is not None:
                result[0] = bytes(sample.payload)
                event.set()

        self._session.get(topic, _on_reply, payload=raw, timeout=timeout)
        # Wait slightly longer than zenoh's own timeout so zenoh fires first.
        _ = event.wait(timeout + 0.5)
        if result[0] is None:
            logger.warning('query %r timed out after %.1fs', topic, timeout)
        else:
            logger.debug('query %r -> %d bytes', topic, len(result[0]))
        return result[0]
