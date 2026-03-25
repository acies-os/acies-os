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
    key: list[str]  # path to the value: ['k1', 'k2'] → config['k1']['k2']


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


class AciesRoute(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond
    old_topic: str | None
    new_topic: str | None


# ------------------------------- data messages -------------------------------
# Forwarded to user handlers. Users may also define their own msgspec.Struct
# types for application-specific payloads.


class AciesTensor(msgspec.Struct, frozen=True):
    source: str
    timestamp: NanoSecond
    payload: list[float | int | bytes]
    metadata: dict | None = None
