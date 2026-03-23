"""AciesContext, AppState, TaskState — handler context and shared state.

Handlers interact with the messaging layer exclusively through AciesContext.
They never touch transports or the router directly.

AciesContext depends on no other module in this package — it receives
publish and query capabilities as plain callables injected by AciesApp.
This avoids any circular imports between context, router, and executor.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeAlias

import msgspec

# Matches router.publish / router.query signatures; injected into AciesContext.
Publisher: TypeAlias = Callable[[str, bytes], None]
Querier: TypeAlias = Callable[[str, bytes, float], bytes | None]


def deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge source into target in-place.

    If both target and source have a dict at the same key, recurse.
    Otherwise overwrite the target value with the source value.
    """
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = value


class AppState:
    """Shared state across all handlers in an app.

    config — nested dict; externally controllable via AciesGet/Set.
             The 'sys' key is reserved for middleware (host, name, etc.).
    data   — free-form transient state; internal to the app, not externally controlled.
    """

    def __init__(self) -> None:
        self.lock: threading.RLock = threading.RLock()
        self.config: dict[str, Any] = {}
        self.data: dict[str, Any] = {}


class TaskState:
    """Per-task state, shared across all jobs of the same task."""

    def __init__(self) -> None:
        self.lock: threading.RLock = threading.RLock()
        self.data: dict[str, Any] = {}


class AciesContext:
    def __init__(
        self,
        publish_fn: Publisher,
        query_fn: Querier,
        app: AppState,
        task: TaskState,
    ) -> None:
        self._publish_fn: Publisher = publish_fn
        self._query_fn: Querier = query_fn
        self.app: AppState = app
        self.task: TaskState = task

    def publish(self, topic: str, msg: msgspec.Struct) -> None:
        """Encode msg and publish raw bytes to topic.

        Encoding happens here (worker thread) so the router stays byte-only.
        """
        self._publish_fn(topic, msgspec.msgpack.encode(msg))

    def query(self, topic: str, msg: msgspec.Struct, timeout: float = 1.0) -> msgspec.Struct | None:
        """Synchronous RPC. Encodes the request, blocks until a reply arrives
        or timeout expires, then decodes and returns the reply struct.

        Returns None on timeout.
        """
        raw = self._query_fn(topic, msgspec.msgpack.encode(msg), timeout)
        if raw is None:
            return None
        # TODO: decode with the reply type once reply typing is wired through specs
        return msgspec.msgpack.decode(raw)
