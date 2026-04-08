"""AciesContext, AppState, TaskState — handler context and shared state.

Handlers interact with the messaging layer exclusively through AciesContext.
They never touch transports or the router directly.

AciesContext depends on no other module in this package — it receives
publish and query capabilities as plain callables injected by AciesApp.
This avoids any circular imports between context, router, and executor.

Ergonomic access:

    ctx['key']         -> ctx.task.data['key']
    ctx.app['key']     -> ctx.app.data['key']
    ctx.cfg['key']     -> ctx.app.config['key']
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

import msgspec

from .msg import NanoSecond
from .namespace import Namespace

# Matches router.publish / router.query / time.time_ns signatures; injected into AciesContext.
Publisher: TypeAlias = Callable[[str, bytes], None]
Querier: TypeAlias = Callable[[str, bytes, float], bytes | None]
NowFn: TypeAlias = Callable[[], NanoSecond]


def deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge source into target in-place.

    If both target and source have a dict at the same key, recurse.
    Otherwise overwrite the target value with the source value.
    """
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)  # pyright: ignore[reportUnknownArgumentType]
        else:
            target[key] = value


@dataclass
class AppState:
    """Shared state across all handlers in an app.

    config — nested dict; externally controllable via AciesGet/Set.
             The 'sys' key is reserved for middleware (host, name, etc.).
    data   — free-form transient state; internal to the app, not externally controlled.

    Supports dict-style access as a shorthand for .data:

        app['model'] = x      # app.data['model'] = x
        app['model']          # app.data['model']

    Thread safety: handlers run concurrently on worker threads. Use ``app.lock``
    when reading and writing multiple keys as an atomic unit. Single-key reads
    and writes on CPython are effectively atomic due to the GIL, but compound
    operations (read-modify-write) require explicit locking::

        with ctx.app.lock:
            ctx.app['count'] += 1
    """

    lock: threading.RLock = field(default_factory=threading.RLock)
    config: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class TaskState:
    """Per-task state, shared across all jobs of the same task.

    Supports dict-style access as a shorthand for .data:

        task['buf'] = x       # task.data['buf'] = x
        task['buf']           # task.data['buf']

    Thread safety: if a task's handler can be dispatched concurrently (e.g. a
    subscriber with a busy queue), use ``task.lock`` to guard compound
    operations on .data.
    """

    lock: threading.RLock = field(default_factory=threading.RLock)
    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __contains__(self, key: object) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class AciesContext:
    def __init__(
        self,
        publish_fn: Publisher,
        query_fn: Querier,
        now_fn: NowFn,
        app: AppState,
        task: TaskState,
        ns: Namespace,
    ) -> None:
        self._publish_fn: Publisher = publish_fn
        self._query_fn: Querier = query_fn
        self._now_fn: NowFn = now_fn
        self.app: AppState = app
        self.task: TaskState = task
        self.ns: Namespace = ns

    @property
    def cfg(self) -> dict[str, Any]:
        """Shorthand for ctx.app.config."""
        return self.app.config

    def __getitem__(self, key: str) -> Any:
        return self.task[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.task[key] = value

    def __delitem__(self, key: str) -> None:
        del self.task[key]

    def __contains__(self, key: object) -> bool:
        return key in self.task

    def get(self, key: str, default: Any = None) -> Any:
        return self.task.get(key, default)

    def now(self) -> NanoSecond:
        """Return current time in nanoseconds. Mockable in tests via now_fn injection."""
        return self._now_fn()

    def publish(self, topic: str, msg: Any) -> None:
        """Encode msg and publish raw bytes to topic.

        Encoding happens here (worker thread) so the router stays byte-only.
        """
        self._publish_fn(topic, msgspec.msgpack.encode(msg))

    def query(
        self,
        topic: str,
        msg: msgspec.Struct,
        timeout: float = 1.0,
        reply_type: type = dict,
    ) -> msgspec.Struct | None:
        """Synchronous RPC. Encodes the request, blocks until a reply arrives
        or timeout expires, then decodes and returns the reply struct.

        Returns None on timeout.
        """
        raw = self._query_fn(topic, msgspec.msgpack.encode(msg), timeout)
        if raw is None:
            return None
        return msgspec.msgpack.decode(raw, type=reply_type)
