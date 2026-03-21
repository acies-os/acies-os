"""Transport protocol and LocalTransport.

Transport is a Protocol — backends don't need to inherit from it, they just
need to implement the required methods. The type checker enforces correctness
structurally, and new backends can be added anywhere in the repo without
touching this file.

Routing between transports is prefix-based: the Router calls can_handle()
on each transport to decide where to send outbound messages.

Concrete backends:
  ZenohTransport     — cross-node pub/sub + queryable; handles bare topics (default)
  LocalTransport     — in-process queue; used for tests (implemented here)
  WebSocketTransport — browser/UI; topic prefix 'ws://'

on_message callback convention
-------------------------------
The callback passed to start() has signature:

    on_message(topic: str, raw: bytes, reply_fn: Callable | None) -> None

reply_fn is None for regular pub messages. For queries, it is a callable the
executor will invoke with the handler's return value; it encodes the result and
sends it back to the waiting query() caller.
"""

from __future__ import annotations

import queue
import re
import threading
from typing import Any, Callable, Protocol, TypeAlias

# Callable that sends encoded reply bytes back to a waiting query() caller.
SendBytes: TypeAlias = Callable[[bytes], None]

# Callable the executor invokes with the handler's return value.
# Encodes the result and calls SendBytes to unblock the query() caller.
ReplyFn: TypeAlias = Callable[[Any], None]

# Factory called once per incoming query to produce a ReplyFn.
# The transport supplies the SendBytes closure; the factory wires it to the
# executor's call convention (encode result → send bytes).
ReplyFnFactory: TypeAlias = Callable[[SendBytes], ReplyFn]


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
    def start(self, on_message: Callable[[str, bytes, ReplyFn | None], None]) -> None:
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

    def advertise(self, topic: str, reply_fn_factory: ReplyFnFactory) -> None:
        """Register a queryable endpoint.

        reply_fn_factory(send_bytes) -> reply_fn
          send_bytes  — transport-provided closure that delivers the encoded reply
          reply_fn    — callable the executor will call with the handler's return value
        """
        ...


class _Sentinel:
    pass


class LocalTransport:
    """In-process queue-based transport for testing and single-process apps.

    Always handles all topics (can_handle returns True). Add to Router last so
    prefixed transports like WebSocketTransport take priority.

    Query flow
    ----------
    1. Caller calls query(topic, raw, timeout).
    2. LocalTransport finds the registered reply_fn_factory for the topic.
    3. Creates a send_bytes closure backed by a threading.Event.
    4. Calls factory(send_bytes) to produce a reply_fn the executor will invoke.
    5. Puts (topic, raw, reply_fn) into the inbound queue.
    6. Receiver thread delivers it via on_message(topic, raw, reply_fn).
    7. Router creates a Job with that reply_fn; executor calls reply_fn(result).
    8. reply_fn encodes result and calls send_bytes(encoded), which sets the event.
    9. query() unblocks and returns the encoded bytes.
    """

    _SENTINEL: _Sentinel = _Sentinel()

    def __init__(self) -> None:
        self._subscriptions: set[str] = set()
        self._advertisers: dict[str, ReplyFnFactory] = {}
        self._queue: queue.SimpleQueue[tuple[str, bytes, ReplyFn | None] | _Sentinel] = queue.SimpleQueue()
        self._on_message: Callable[[str, bytes, ReplyFn | None], None] | None = None
        self._thread: threading.Thread | None = None

    def start(self, on_message: Callable[[str, bytes, ReplyFn | None], None]) -> None:
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

    def advertise(self, topic: str, reply_fn_factory: ReplyFnFactory) -> None:
        self._advertisers[topic] = reply_fn_factory

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Send a query and block until a reply arrives or timeout elapses."""
        factory = self._find_advertiser(topic)
        event = threading.Event()
        result: list[bytes | None] = [None]

        if factory is not None:

            def send_bytes(b: bytes) -> None:
                result[0] = b
                event.set()

            reply_fn: ReplyFn = factory(send_bytes)
            self._queue.put((topic, raw, reply_fn))

        _ = event.wait(timeout)
        return result[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_advertiser(self, topic: str) -> ReplyFnFactory | None:
        for pattern, factory in self._advertisers.items():
            if _topic_matches(pattern, topic):
                return factory
        return None

    def _receiver_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            assert isinstance(item, tuple)
            topic, raw, reply_fn = item
            if self._on_message is not None:
                self._on_message(topic, raw, reply_fn)
