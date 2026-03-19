"""Transport ABC and LocalTransport.

Transport is the base class for all messaging backends. Each backend owns
its receiver thread(s) and calls on_message when a message arrives, pushing
it into the router's inbound queue.

Routing between transports is prefix-based: the Router calls can_handle()
on each transport to decide where to send outbound messages.

Concrete backends (to be implemented):
  ZenohTransport   — cross-node pub/sub + queryable; handles bare topics (default)
  LocalTransport   — in-process queue; used for tests (implemented here)
  WebSocketTransport — browser/UI; topic prefix 'ws://'
  IPCTransport     — inter-process on same machine; topic prefix 'ipc://'
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .msg import AciesMsg


class Transport(ABC):
    @abstractmethod
    def start(self, on_message: Callable[[str, 'AciesMsg'], None]) -> None:
        """Start receiver thread(s). Call on_message(topic, msg) on each arrival."""

    @abstractmethod
    def stop(self) -> None:
        """Stop receiver thread(s) and release resources."""

    @abstractmethod
    def can_handle(self, topic: str) -> bool:
        """Return True if this transport should handle the given topic."""

    @abstractmethod
    def publish(self, topic: str, msg: 'AciesMsg') -> None:
        """Send a message. Called synchronously from worker threads."""

    @abstractmethod
    def subscribe(self, topic: str) -> None:
        """Register interest in a topic so the receiver thread delivers it."""

    @abstractmethod
    def query(self, topic: str, msg: 'AciesMsg', timeout: float) -> 'AciesMsg | None':
        """Synchronous RPC call. Blocks until reply or timeout."""

    @abstractmethod
    def advertise(self, topic: str, reply_fn_factory: Callable) -> None:
        """Register a queryable endpoint. reply_fn_factory creates the reply_fn
        closure passed into the Job so the executor can reply after the handler."""


class LocalTransport(Transport):
    """In-process queue-based transport for testing and single-process apps.

    Handles all topics (no prefix required). Full implementation in Phase 2.
    """

    def start(self, on_message: Callable[[str, 'AciesMsg'], None]) -> None:
        # TODO: Phase 2 — start receiver thread, store on_message callback
        ...

    def stop(self) -> None:
        # TODO: Phase 2
        ...

    def can_handle(self, topic: str) -> bool:
        return True  # handles all topics; add to Router last so prefixed transports take priority

    def publish(self, topic: str, msg: 'AciesMsg') -> None:
        # TODO: Phase 2 — deliver directly to subscribers in-process
        ...

    def subscribe(self, topic: str) -> None:
        # TODO: Phase 2
        ...

    def query(self, topic: str, msg: 'AciesMsg', timeout: float) -> 'AciesMsg | None':
        # TODO: Phase 2
        ...

    def advertise(self, topic: str, reply_fn_factory: Callable) -> None:
        # TODO: Phase 2
        ...
