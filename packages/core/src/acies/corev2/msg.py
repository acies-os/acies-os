"""AciesMsg — wire message format.

Placeholder for Phase 1. Full implementation deferred until runtime needs
(serialization format, transport encoding) are clearer in Phase 2.
"""

from __future__ import annotations

from typing import Any


class AciesMsg:
    """Placeholder. To be fully designed in Phase 2."""

    def __init__(self, payload: Any = None, metadata: dict | None = None, topic: str = '') -> None:
        self.payload = payload
        self.metadata = metadata or {}
        self.topic = topic

    def __repr__(self) -> str:
        return f'AciesMsg(payload={self.payload!r})'
