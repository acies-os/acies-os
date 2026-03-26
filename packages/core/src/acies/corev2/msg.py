"""Typed message structs for AciesOS.

Two categories:
- Control messages: middleware-internal, used by default control handlers.
- Data messages: forwarded to user handlers as typed msgspec.Struct objects.

Wire encoding:
- ZenohTransport: MessagePack (msgspec.msgpack.encode/decode)
- WebSocketTransport: JSON (msgspec.json.encode/decode)
- LocalTransport: no encoding; typed objects passed directly
"""

from __future__ import annotations

from typing import Any, TypeAlias

import msgspec

NanoSecond: TypeAlias = int  # nanoseconds since Unix epoch (time.time_ns())

# ---------------------------- Result type (Ok/Err) ----------------------------


class Ok(msgspec.Struct, frozen=True, tag=True, tag_field='type'):
    value: Any = None  # populated for Get; None for Set/Del


class Err(msgspec.Struct, frozen=True, tag=True, tag_field='type'):
    reason: str  # 'key_not_found' | 'key_protected' | 'invalid_path'


AciesResult: TypeAlias = Ok | Err


# --------------------------------- kv ops ------------------------------------
# Used in AciesKvRequest.ops — one entry per key path operation.


class AciesGet(msgspec.Struct, frozen=True, tag=True, tag_field='type'):
    key: list[str]  # path to the value: ['k1', 'k2'] -> config['k1']['k2']


class AciesSet(msgspec.Struct, frozen=True, tag=True, tag_field='type'):
    key: list[str]
    value: Any


class AciesDel(msgspec.Struct, frozen=True, tag=True, tag_field='type'):
    key: list[str]


KvEntry: TypeAlias = AciesGet | AciesSet | AciesDel


# ------------------------------ control messages ------------------------------


class AciesHeartbeat(msgspec.Struct, frozen=True):
    source: str
    state: str
    timestamp: NanoSecond


class AciesKvRequest(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond
    ops: list[KvEntry]


class AciesKvResponse(msgspec.Struct, frozen=True):
    timestamp: NanoSecond
    results: list[AciesResult]


class TopicRename(msgspec.Struct, frozen=True):
    old: str | None = None  # None = add only (no unsubscribe)
    new: str | None = None  # None = remove only (no subscribe)

    def __post_init__(self):
        if self.old is None and self.new is None:
            raise ValueError('at least one of `old` or `new` must be set')


class AciesRouteRequest(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond
    spec_id: str | None = None  # preferred: UUID from spec.id
    spec_name: str | None = None  # fallback: match by name
    inputs: list[TopicRename] = []
    outputs: list[TopicRename] = []

    def __post_init__(self):
        if self.spec_id is None and self.spec_name is None:
            raise ValueError('at least one of `spec_id` or `spec_name` must be set')


class AciesRouteResponse(msgspec.Struct, frozen=True):
    timestamp: NanoSecond
    result: AciesResult


class AciesIoRequest(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond


class AciesIoResponse(msgspec.Struct, frozen=True):
    timestamp: NanoSecond
    io: dict[str, dict[str, str | list[str]]]


class AciesSchemaRequest(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond


class AciesSchemaResponse(msgspec.Struct, frozen=True):
    timestamp: NanoSecond
    schemas: dict[str, Any]  # keyed by spec.id; values are schema entry dicts


# ------------------------------- data messages -------------------------------
# Forwarded to user handlers. Users may also define their own msgspec.Struct
# types for application-specific payloads.


class AciesTensor(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond
    payload: list[float | int | bytes]
    metadata: dict | None = None
