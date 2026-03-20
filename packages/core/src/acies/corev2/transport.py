"""Transport protocol and LocalTransport.

Transport is a Protocol — backends don't need to inherit from it, they just
need to implement the required methods. The type checker enforces correctness
structurally, and new backends can be added anywhere in the repo without
touching this file.

Routing between transports is prefix-based: the Router calls can_handle()
on each transport to decide where to send outbound messages.

Concrete backends (to be implemented):
  ZenohTransport     — cross-node pub/sub + queryable; handles bare topics (default)
  LocalTransport     — in-process queue; used for tests (implemented here)
  WebSocketTransport — browser/UI; topic prefix 'ws://'
  IPCTransport       — inter-process on same machine; topic prefix 'ipc://'
"""

from __future__ import annotations

from typing import Callable, Protocol


class Transport(Protocol):
    def start(self, on_message: Callable[[str, bytes], None]) -> None:
        """Start receiver thread(s). Call on_message(topic, raw_bytes) on each arrival."""
        ...

    def stop(self) -> None:
        """Stop receiver thread(s) and release resources."""
        ...

    def can_handle(self, topic: str) -> bool:
        """Return True if this transport should handle the given topic."""
        ...

    def publish(self, topic: str, raw: bytes) -> None:
        """Send raw bytes. Called synchronously from the router."""
        ...

    def subscribe(self, topic: str) -> None:
        """Register interest in a topic so the receiver thread delivers it."""
        ...

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Synchronous RPC call. Blocks until reply bytes arrive or timeout."""
        ...

    def advertise(self, topic: str, reply_fn_factory: Callable[..., None]) -> None:
        """Register a queryable endpoint. reply_fn_factory creates the reply_fn
        closure passed into the Job so the executor can reply after the handler."""
        ...


class LocalTransport:
    """In-process queue-based transport for testing and single-process apps.

    Handles all topics (no prefix required). Add to Router last so prefixed
    transports like WebSocketTransport take priority.

    Full implementation in Phase 2.
    """

    def start(self, on_message: Callable[[str, bytes], None]) -> None:
        # TODO: Phase 2 — start receiver thread, store on_message callback
        ...

    def stop(self) -> None:
        # TODO: Phase 2
        ...

    def can_handle(self, topic: str) -> bool:
        return True

    def publish(self, topic: str, raw: bytes) -> None:
        # TODO: Phase 2 — deliver directly to subscribers in-process
        ...

    def subscribe(self, topic: str) -> None:
        # TODO: Phase 2
        ...

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        # TODO: Phase 2
        ...

    def advertise(self, topic: str, reply_fn_factory: Callable[..., None]) -> None:
        # TODO: Phase 2
        ...
