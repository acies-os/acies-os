"""AciesContext, AppState, TaskState — handler context and shared state.

Handlers interact with the messaging layer exclusively through AciesContext.
They never touch transports or the router directly.

AciesContext depends on no other module in this package — it receives
publish and query capabilities as plain callables injected by AciesApp.
This avoids any circular imports between context, router, and executor.
"""

from __future__ import annotations

import threading
from typing import Callable

import msgspec


class AppState:
    """Shared state across all handlers in an app."""

    def __init__(self) -> None:
        self.lock: threading.RLock = threading.RLock()
        self.data: dict = {}


class TaskState:
    """Per-task state, shared across all jobs of the same task."""

    def __init__(self) -> None:
        self.lock: threading.RLock = threading.RLock()
        self.data: dict = {}


class AciesContext:
    def __init__(
        self,
        publish_fn: Callable[..., None],
        query_fn: Callable[..., None],
        app: AppState,
        task: TaskState,
    ) -> None:
        self._publish_fn: Callable[..., None] = publish_fn
        self._query_fn: Callable[..., None] = query_fn
        self.app: AppState = app
        self.task: TaskState = task

    def publish(self, topic: str, msg: msgspec.Struct) -> None:
        """Publish a typed msgspec.Struct to a topic. Synchronous — returns
        after the router has accepted the message for encoding and dispatch."""
        # TODO: Phase 2 — call self._publish_fn(topic, msg)
        ...

    def query(self, topic: str, msg: msgspec.Struct, timeout: float = 1.0) -> msgspec.Struct | None:
        """Synchronous RPC. Blocks until a reply arrives or timeout expires.
        Returns the decoded reply struct, or None on timeout."""
        # TODO: Phase 2 — call self._query_fn(topic, msg, timeout)
        ...
