"""Default system control handlers — registered automatically by AciesApp.

Provides:
  - Heartbeat: periodic pub on ns.ctl.heartbeat; interval default 5 seconds.
  - KV:        queryable at ns.ctl.kv; get/set/del on app.state.config.
               The 'sys' key is protected — set/del on it is rejected.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .context import AciesContext
from .msg import (
    AciesDel,
    AciesGet,
    AciesHeartbeat,
    AciesIoRequest,
    AciesIoResponse,
    AciesKvChange,
    AciesKvRequest,
    AciesKvResponse,
    AciesResult,
    AciesRouteRequest,
    AciesRouteResponse,
    AciesSchemaRequest,
    AciesSchemaResponse,
    AciesSet,
    Err,
    Ok,
)
from .namespace import CtlTopic
from .task import ScheduleSpec, ServiceSpec, SubscriberSpec

if TYPE_CHECKING:
    from .router import Router

_DEFAULT_HEARTBEAT_INTERVAL: float = 5.0

logger = logging.getLogger(__name__)


# --------------------------------- heartbeat ---------------------------------


def _heartbeat(ctx: AciesContext) -> None:
    logger.debug('publishing heartbeat')
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


def _get_path(config: dict[str, Any], path: Sequence[str]) -> Any:
    node: Any = config
    for key in path:
        if not isinstance(node, dict):
            raise KeyError(key)
        node = node[key]  # pyright: ignore[reportUnknownVariableType]
    return node  # pyright: ignore[reportUnknownVariableType]


def _set_path(config: dict[str, Any], path: Sequence[str], value: Any) -> None:
    node = config
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(key)
        node = node[key]
    if path[-1] not in node:
        raise KeyError(path[-1])
    node[path[-1]] = value


def _del_path(config: dict[str, Any], path: Sequence[str]) -> None:
    node: Any = config
    for key in path[:-1]:
        if not isinstance(node, dict):
            raise KeyError(key)
        node = node[key]  # pyright: ignore[reportUnknownVariableType]
    del node[path[-1]]


# sys keys that may be updated at runtime via ctl/kv
_SYS_MUTABLE: frozenset[str] = frozenset({'state'})


def _handle_get(config: dict[str, Any], k: Sequence[str]) -> AciesResult:
    if not k:
        return Err(reason='empty key path')
    try:
        return Ok(value=_get_path(config, k))
    except KeyError as e:
        return Err(reason=f'key not found: {e.args[0]!r}')


def _handle_set(config: dict[str, Any], k: Sequence[str], v: Any) -> AciesResult:
    if not k:
        return Err(reason='empty key path')
    if k[0] == 'sys' and not (len(k) == 2 and k[1] in _SYS_MUTABLE):
        return Err(reason='key_protected')
    try:
        _set_path(config, k, v)
        return Ok()
    except KeyError as e:
        return Err(reason=f'key not found: {e.args[0]!r}')


def _handle_del(config: dict[str, Any], k: Sequence[str]) -> AciesResult:
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
    notifications: list[AciesKvChange] = []
    with ctx.app.lock:
        results: list[AciesResult] = []
        for entry in msg.ops:
            match entry:
                case AciesGet(key=k):
                    results.append(_handle_get(ctx.app.config, k))
                case AciesSet(key=k, value=v):
                    result = _handle_set(ctx.app.config, k, v)
                    results.append(result)
                    if isinstance(result, Ok):
                        notifications.append(AciesKvChange(key=k, op='set', value=v))
                case AciesDel(key=k):
                    result = _handle_del(ctx.app.config, k)
                    results.append(result)
                    if isinstance(result, Ok):
                        notifications.append(AciesKvChange(key=k, op='del'))
    # Publish notifications outside the lock to avoid holding it during I/O.
    for note in notifications:
        topic = f'{ctx.ns.ctl.notify}/{note.key[0]}'
        ctx.publish(topic, note)
    return AciesKvResponse(timestamp=ctx.now(), results=results)


def make_kv_spec() -> ServiceSpec:
    """Return a ServiceSpec for the ctl/kv queryable."""
    return ServiceSpec(name='_kv', fn=_kv, topic=CtlTopic('kv'), msg_type=AciesKvRequest, return_type=AciesKvResponse)


# ---------------------------------- route ------------------------------------


def make_route_spec(router: Router) -> ServiceSpec:
    """Return a ServiceSpec for the ctl/route queryable.

    The handler remaps input topics for a spec identified by id or name.
    For each TopicRename: unsubscribes old topic (if set), subscribes new
    topic (if set). Services are unadvertised/re-advertised the same way.
    """

    def _route(ctx: AciesContext, msg: AciesRouteRequest) -> AciesRouteResponse:
        spec = router.find_spec(msg.spec_id, msg.spec_name)
        if spec is None:
            return AciesRouteResponse(
                timestamp=ctx.now(),
                result=Err(reason=f'spec not found: id={msg.spec_id!r} name={msg.spec_name!r}'),
            )

        for rename in msg.inputs:
            if rename.old is not None:
                if isinstance(spec, SubscriberSpec):
                    router.unsubscribe(rename.old, spec)
                else:
                    router.unadvertise(rename.old)
            if rename.new is not None:
                if isinstance(spec, SubscriberSpec):
                    router.subscribe(rename.new, spec)
                else:
                    assert isinstance(spec, ServiceSpec)
                    router.advertise(rename.new, spec)

        for rename in msg.outputs:
            router.remap_output(spec, rename)

        return AciesRouteResponse(timestamp=ctx.now(), result=Ok())

    return ServiceSpec(
        name='_route', fn=_route, topic=CtlTopic('route'), msg_type=AciesRouteRequest, return_type=AciesRouteResponse
    )


# ------------------------------------ io -------------------------------------


def make_io_spec(router: Router) -> ServiceSpec:
    """Return a ServiceSpec for the ctl/io queryable.

    Returns a snapshot of the router's I/O routing table: which topics each
    spec subscribes to (inputs) and which topics it has published to (outputs).
    """

    def _io(ctx: AciesContext, msg: AciesIoRequest) -> AciesIoResponse:
        logger.debug('I/O routing table requested by %s', msg.source)
        return AciesIoResponse(timestamp=ctx.now(), io=router.io_map)

    return ServiceSpec(name='_io', fn=_io, topic=CtlTopic('io'), msg_type=AciesIoRequest, return_type=AciesIoResponse)


# ---------------------------------- schema -----------------------------------


def make_schema_spec() -> ServiceSpec:
    """Return a ServiceSpec for the ctl/schema queryable.

    Returns the service schema table stored in sys.schemas at startup.
    Each entry contains the resolved topic, request schema, and response
    schema for a registered ServiceSpec.
    """

    def _schema(ctx: AciesContext, msg: AciesSchemaRequest) -> AciesSchemaResponse:
        with ctx.app.lock:
            schemas = dict(ctx.app.config.get('sys', {}).get('schemas', {}))
        logger.debug('service schema requested by %s', msg.source)
        return AciesSchemaResponse(timestamp=ctx.now(), schemas=schemas)

    return ServiceSpec(
        name='_schema',
        fn=_schema,
        topic=CtlTopic('schema'),
        msg_type=AciesSchemaRequest,
        return_type=AciesSchemaResponse,
    )
