"""Transport protocol and shared type aliases.

Transport is a Protocol -- backends don't need to inherit from it, they just
need to implement the required methods. The type checker enforces correctness
structurally, and new backends can be added anywhere in the repo without
touching this file.

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

from typing import Callable, Protocol, TypeAlias

# Delivers encoded reply bytes back to a waiting query() caller.
ReplyCallback: TypeAlias = Callable[[bytes], None]

# Callback passed to Transport.start(); called on every inbound message.
# reply_callback is None for pub messages, set for incoming queries.
MessageHandler: TypeAlias = Callable[[str, bytes, ReplyCallback | None], None]


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

    def unsubscribe(self, topic: str) -> None:
        """Deregister interest in a topic."""
        ...

    def unadvertise(self, topic: str) -> None:
        """Deregister a queryable endpoint."""
        ...
