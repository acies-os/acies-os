"""Transport protocol, LocalTransport, and ZenohTransport.

Transport is a Protocol — backends don't need to inherit from it, they just
need to implement the required methods. The type checker enforces correctness
structurally, and new backends can be added anywhere in the repo without
touching this file.

Routing between transports is prefix-based: the Router uses prefix rules
to decide which transport handles a given topic.

Concrete backends:
  ZenohTransport     — cross-node pub/sub + queryable; handles bare topics (default)
  LocalTransport     — in-process queue; used for tests
  WebSocketTransport — browser/UI; topic prefix 'ws://'

on_message callback convention
-------------------------------
The callback passed to start() has signature:

    on_message(topic: str, raw: bytes, reply_fn: ReplyCallback | None) -> None

reply_fn is None for regular pub messages. For queries, it is a
transport-owned closure that delivers encoded reply bytes back to the waiting
query() caller (e.g. sets a threading.Event). The dispatch function encodes
the handler's return value and calls reply_fn(encoded_bytes) to complete
the reply.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Callable, Protocol, TypeAlias

import zenoh

from ._concurrency import SENTINEL, Sentinel

# Delivers encoded reply bytes back to a waiting query() caller.
ReplyCallback: TypeAlias = Callable[[bytes], None]

# Callback passed to Transport.start(); called on every inbound message.
# reply_callback is None for pub messages, set for incoming queries.
MessageHandler: TypeAlias = Callable[[str, bytes, ReplyCallback | None], None]


def _topic_matches(pattern: str, topic: str) -> bool:
    """Return True if topic matches pattern.

    Supports zenoh-style wildcards:
      *   — exactly one chunk (non-empty sequence of non-'/' chars)
      **  — any number of chunks, including zero (may span multiple '/' separators)
    Exact match always works.
    """
    if pattern == topic:
        return True
    regex = ''
    i = 0
    while i < len(pattern):
        if pattern[i : i + 2] == '**':
            regex += '.*'
            i += 2
        elif pattern[i] == '*':
            regex += '[^/]+'
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return bool(re.fullmatch(regex, topic))


class Transport(Protocol):
    def start(self, on_message: MessageHandler) -> None:
        """Start receiver thread(s).

        Calls on_message(topic, raw_bytes, reply_fn) on each arrival.
        reply_fn is None for pub messages; set for incoming queries.
        """
        ...

    def stop(self) -> None:
        """Graceful shutdown: finish delivering already-queued messages, then stop."""
        ...

    def abort(self) -> None:
        """Immediate shutdown: discard queued messages and stop."""
        ...

    def publish(self, topic: str, raw: bytes) -> None:
        """Send raw bytes to all matching subscribers."""
        ...

    def subscribe(self, topic: str) -> None:
        """Register interest in a topic so the receiver thread delivers it."""
        ...

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Synchronous RPC call. Blocks until reply bytes arrive or timeout."""
        ...

    def advertise(self, topic: str) -> None:
        """Register a queryable endpoint.

        Marks topic as queryable. When a query arrives, the transport creates
        a reply_fn closure for that specific caller and passes it as the
        third argument to on_message. The dispatch function encodes the
        handler result and calls reply_fn(encoded) to deliver the reply.
        """
        ...


class LocalTransport:
    """In-process queue-based transport for testing and single-process apps.

    Always handles all topics. Add to Router last so prefixed transports like
    WebSocketTransport take priority.

    Query flow
    ----------
    1. Caller calls query(topic, raw, timeout).
    2. LocalTransport checks that the topic is advertised.
    3. Creates a reply_fn closure backed by a threading.Event.
    4. Puts (topic, raw, reply_fn) directly into the inbound queue.
    5. Receiver thread delivers it via on_message(topic, raw, reply_fn).
    6. Router creates a Job with reply_fn; dispatch encodes the result and
       calls job.reply_fn(encoded_bytes).
    7. reply_fn sets result[0] and the event; query() unblocks and returns
       the encoded bytes.
    """

    def __init__(self) -> None:
        self._subscriptions: set[str] = set()
        self._advertisers: set[str] = set()
        self._queue: queue.SimpleQueue[tuple[str, bytes, ReplyCallback | None] | Sentinel] = queue.SimpleQueue()
        self._on_message: MessageHandler | None = None
        self._thread: threading.Thread | None = None

    def start(self, on_message: MessageHandler) -> None:
        self._on_message = on_message
        self._thread = threading.Thread(target=self._receiver_loop, name='local-transport', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(SENTINEL)
        if self._thread:
            self._thread.join()

    def abort(self) -> None:
        while True:
            try:
                _ = self._queue.get(block=False)
            except queue.Empty:
                break
        self._queue.put(SENTINEL)
        if self._thread:
            self._thread.join()

    def subscribe(self, topic: str) -> None:
        self._subscriptions.add(topic)

    def publish(self, topic: str, raw: bytes) -> None:
        """Deliver raw bytes if topic matches any active subscription."""
        if any(_topic_matches(pattern, topic) for pattern in self._subscriptions):
            self._queue.put((topic, raw, None))

    def advertise(self, topic: str) -> None:
        self._advertisers.add(topic)

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Send a query and block until a reply arrives or timeout elapses."""
        event = threading.Event()
        result: list[bytes | None] = [None]

        if self._is_advertised(topic):

            def reply_fn(b: bytes) -> None:
                result[0] = b
                event.set()

            self._queue.put((topic, raw, reply_fn))

        _ = event.wait(timeout)
        return result[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_advertised(self, topic: str) -> bool:
        return any(_topic_matches(pattern, topic) for pattern in self._advertisers)

    def _receiver_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is SENTINEL:
                break
            assert isinstance(item, tuple)
            topic, raw, reply_fn = item
            if self._on_message is not None:
                self._on_message(topic, raw, reply_fn)


class ZenohTransport:
    """Zenoh-backed transport — the primary production backend.

    Each AciesApp holds one ZenohTransport. Cross-node and cross-process
    routing is handled transparently by the zenoh network. Same-host IPC
    uses UDP multicast discovery (no daemon required); for explicit
    endpoints pass a custom zenoh.Config.

    Args:
        config: Optional zenoh.Config. Defaults to zenoh.Config() (peer
                mode, UDP multicast discovery).
    """

    def __init__(self, config: zenoh.Config | None = None) -> None:
        self._config: zenoh.Config = config if config is not None else zenoh.Config()
        self._session: zenoh.Session | None = None
        self._on_message: MessageHandler | None = None
        self._subscribers: list[zenoh.Subscriber[None]] = []
        self._queryables: list[zenoh.Queryable[None]] = []

    def start(self, on_message: MessageHandler) -> None:
        """Open the zenoh session and store the inbound message callback."""
        self._on_message = on_message
        self._session = zenoh.open(self._config)

    def stop(self) -> None:
        """Undeclare all subscribers/queryables and close the session."""
        for sub in self._subscribers:
            sub.undeclare()  # pyright: ignore[reportUnknownMemberType]
        self._subscribers.clear()
        for qb in self._queryables:
            qb.undeclare()  # pyright: ignore[reportUnknownMemberType]
        self._queryables.clear()
        if self._session is not None:
            self._session.close()  # pyright: ignore[reportUnknownMemberType]
            self._session = None

    def abort(self) -> None:
        """Immediate shutdown — same as stop() for zenoh."""
        self.stop()

    def subscribe(self, topic: str) -> None:
        """Declare a zenoh subscriber that forwards samples to on_message."""
        assert self._session is not None, 'call start() before subscribe()'

        def _on_sample(sample: zenoh.Sample) -> None:
            assert self._on_message is not None, 'on_message callback must be set before subscribing'
            self._on_message(str(sample.key_expr), bytes(sample.payload), None)

        self._subscribers.append(self._session.declare_subscriber(topic, _on_sample))

    def publish(self, topic: str, raw: bytes) -> None:
        """Put raw bytes to topic."""
        assert self._session is not None, 'call start() before publish()'
        self._session.put(topic, raw)  # pyright: ignore[reportUnknownMemberType]

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

        self._queryables.append(self._session.declare_queryable(topic, _on_query))

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
        return result[0]
