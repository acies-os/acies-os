"""Default system control handlers — registered automatically by AciesApp.

Provides:
  - Heartbeat: periodic pub on ns.ctl.heartbeat; interval default 5 seconds.
  - KV:        queryable at ns.ctl.kv; get/set/del on app.state.config.
               The 'sys' key is protected — set/del on it is rejected.
"""

from __future__ import annotations

from typing import Any

from .context import AciesContext
from .msg import (
    AciesHeartbeat,
    AciesKvRequest,
    AciesKvResponse,
    Del,
    Err,
    Get,
    KvResult,
    Ok,
    Set,
)
from .namespace import CtlTopic
from .task import ScheduleSpec, ServiceSpec

_DEFAULT_HEARTBEAT_INTERVAL: float = 5.0


# --------------------------------- heartbeat ---------------------------------


def _heartbeat(ctx: AciesContext) -> None:
    sys = ctx.app.config.get('sys', {})
    ctx.publish(
        ctx.ns.ctl.heartbeat,
        AciesHeartbeat(
            source=ctx.ns.base,
            state=sys.get('state', 'active'),
            timestamp=ctx.now(),
        ),
    )


def make_heartbeat_spec(interval: float = _DEFAULT_HEARTBEAT_INTERVAL) -> ScheduleSpec:
    """Return a ScheduleSpec that publishes AciesHeartbeat at *interval* seconds."""
    return ScheduleSpec(name='_heartbeat', fn=_heartbeat, interval=interval)


# ---------------------------- KV helper functions ----------------------------


def _get_path(config: dict[str, Any], path: list[str]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict):
            raise KeyError(key)
        node = node[key]  # pyright: ignore[reportUnknownVariableType]
    return node  # pyright: ignore[reportUnknownVariableType]


def _set_path(config: dict[str, Any], path: list[str], value: Any) -> None:
    node = config
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(key)
        node = node[key]
    if path[-1] not in node:
        raise KeyError(path[-1])
    node[path[-1]] = value


def _del_path(config: dict[str, Any], path: list[str]) -> None:
    node: Any = config
    for key in path[:-1]:
        if not isinstance(node, dict):
            raise KeyError(key)
        node = node[key]  # pyright: ignore[reportUnknownVariableType]
    del node[path[-1]]


# sys keys that may be updated at runtime via ctl/kv
_SYS_MUTABLE: frozenset[str] = frozenset({'state'})


def _handle_get(config: dict[str, Any], k: list[str]) -> KvResult:
    if not k:
        return Err(reason='empty key path')
    try:
        return Ok(value=_get_path(config, k))
    except KeyError as e:
        return Err(reason=f'key not found: {e.args[0]!r}')


def _handle_set(config: dict[str, Any], k: list[str], v: Any) -> KvResult:
    if not k:
        return Err(reason='empty key path')
    if k[0] == 'sys' and not (len(k) == 2 and k[1] in _SYS_MUTABLE):
        return Err(reason='key_protected')
    try:
        _set_path(config, k, v)
        return Ok()
    except KeyError as e:
        return Err(reason=f'key not found: {e.args[0]!r}')


def _handle_del(config: dict[str, Any], k: list[str]) -> KvResult:
    if not k:
        return Err(reason='empty key path')
    if k[0] == 'sys':
        return Err(reason='key_protected')
    try:
        _del_path(config, k)
        return Ok()
    except KeyError as e:
        return Err(reason=f'key not found: {e.args[0]!r}')


def _kv(ctx: AciesContext, msg: AciesKvRequest) -> AciesKvResponse:
    with ctx.app.lock:
        results: list[KvResult] = []
        for entry in msg.ops:
            match entry:
                case Get(key=k):
                    results.append(_handle_get(ctx.app.config, k))
                case Set(key=k, value=v):
                    results.append(_handle_set(ctx.app.config, k, v))
                case Del(key=k):
                    results.append(_handle_del(ctx.app.config, k))
    return AciesKvResponse(timestamp=ctx.now(), results=results)


def make_kv_spec() -> ServiceSpec:
    """Return a ServiceSpec for the ctl/kv queryable."""
    return ServiceSpec(name='_kv', fn=_kv, topic=CtlTopic('kv'), msg_type=AciesKvRequest)
