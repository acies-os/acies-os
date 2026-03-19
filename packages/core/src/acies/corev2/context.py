"""AciesContext — injected into every handler.

Handlers interact with the messaging layer exclusively through AciesContext.
They never touch transports or the router directly.

AciesContext depends on no other module in this package — it receives
publish and query capabilities as plain callables injected by AciesApp.
This avoids any circular imports between context, router, and executor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .msg import AciesMsg


class AciesContext:
    def __init__(self, publish_fn: Callable[..., None], query_fn: Callable[..., None]) -> None:
        self._publish_fn: Callable[..., None] = publish_fn
        self._query_fn: Callable[..., None] = query_fn

    def publish(self, topic: str, payload: Any, metadata: dict | None = None) -> None:
        """Publish a message to a topic. Synchronous — returns after the
        transport has accepted the message."""
        # TODO: Phase 2 — build AciesMsg, call self._publish_fn
        ...

    def query(self, topic: str, payload: Any = None, timeout: float = 1.0) -> 'AciesMsg | None':
        """Synchronous RPC. Blocks until a reply arrives or timeout expires."""
        # TODO: Phase 2 — build AciesMsg, call self._query_fn
        ...

    def msg(self, payload: Any, metadata: dict | None = None) -> 'AciesMsg':
        """Helper to build an AciesMsg, e.g. for constructing a reply."""
        # TODO: Phase 2
        ...
