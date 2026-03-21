"""Transport protocol and LocalTransport.

Transport is a Protocol — backends don't need to inherit from it, they just
need to implement the required methods. The type checker enforces correctness
structurally, and new backends can be added anywhere in the repo without
touching this file.

Routing between transports is prefix-based: the Router uses prefix rules
to decide which transport handles a given topic.

Concrete backends:
  ZenohTransport     — cross-node pub/sub + queryable; handles bare topics (default)
  LocalTransport     — in-process queue; used for tests (implemented here)
  WebSocketTransport — browser/UI; topic prefix 'ws://'

on_message callback convention
-------------------------------
The callback passed to start() has signature:

    on_message(topic: str, raw: bytes, send_bytes: SendBytes | None) -> None

send_bytes is None for regular pub messages. For queries, it is a
transport-owned closure that delivers encoded reply bytes back to the waiting
query() caller (e.g. sets a threading.Event). The dispatch function encodes
the handler's return value and calls send_bytes(encoded_bytes) to complete
the reply.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Callable, Protocol, TypeAlias

# Callable that sends encoded reply bytes back to a waiting query() caller.
SendBytes: TypeAlias = Callable[[bytes], None]


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
    def start(self, on_message: Callable[[str, bytes, SendBytes | None], None]) -> None:
        """Start receiver thread(s).

        Calls on_message(topic, raw_bytes, send_bytes) on each arrival.
        send_bytes is None for pub messages; set for incoming queries.
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
        a send_bytes closure for that specific caller and passes it as the
        third argument to on_message. The dispatch function encodes the
        handler result and calls send_bytes(encoded) to deliver the reply.
        """
        ...


class _Sentinel:
    pass


class LocalTransport:
    """In-process queue-based transport for testing and single-process apps.

    Always handles all topics. Add to Router last so prefixed transports like
    WebSocketTransport take priority.

    Query flow
    ----------
    1. Caller calls query(topic, raw, timeout).
    2. LocalTransport checks that the topic is advertised.
    3. Creates a send_bytes closure backed by a threading.Event.
    4. Puts (topic, raw, send_bytes) directly into the inbound queue.
    5. Receiver thread delivers it via on_message(topic, raw, send_bytes).
    6. Router creates a Job with send_bytes; dispatch encodes the result and
       calls job.send_bytes(encoded_bytes).
    7. send_bytes sets result[0] and the event; query() unblocks and returns
       the encoded bytes.
    """

    _SENTINEL: _Sentinel = _Sentinel()

    def __init__(self) -> None:
        self._subscriptions: set[str] = set()
        self._advertisers: set[str] = set()
        self._queue: queue.SimpleQueue[tuple[str, bytes, SendBytes | None] | _Sentinel] = queue.SimpleQueue()
        self._on_message: Callable[[str, bytes, SendBytes | None], None] | None = None
        self._thread: threading.Thread | None = None

    def start(self, on_message: Callable[[str, bytes, SendBytes | None], None]) -> None:
        self._on_message = on_message
        self._thread = threading.Thread(target=self._receiver_loop, name='local-transport', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(self._SENTINEL)
        if self._thread:
            self._thread.join()

    def abort(self) -> None:
        while True:
            try:
                _ = self._queue.get(block=False)
            except queue.Empty:
                break
        self._queue.put(self._SENTINEL)
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

            def send_bytes(b: bytes) -> None:
                result[0] = b
                event.set()

            self._queue.put((topic, raw, send_bytes))

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
            if item is self._SENTINEL:
                break
            assert isinstance(item, tuple)
            topic, raw, reply_fn = item
            if self._on_message is not None:
                self._on_message(topic, raw, reply_fn)
