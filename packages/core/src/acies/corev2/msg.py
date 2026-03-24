"""Typed message structs for AciesOS.

Two categories:
- Control messages: middleware-internal, dispatched by reserved topic prefix
  (acies/ctrl/), never forwarded to user handlers.
- Data messages: forwarded to user handlers as typed msgspec.Struct objects.

Wire encoding:
- ZenohTransport: MessagePack (msgspec.msgpack.encode/decode)
- WebSocketTransport: JSON (msgspec.json.encode/decode)
- LocalTransport: no encoding; typed objects passed directly

Control message transport:
- AciesHeartbeat, AciesRoute: pub/sub (fire-and-forget)
- AciesGet, AciesSet, AciesDelete: Zenoh queryable (request/reply via native
  Zenoh reply mechanism). Router registers queryables for acies/ctrl/get,
  acies/ctrl/set, acies/ctrl/delete on startup; handles them internally and
  never creates Jobs.
"""

from __future__ import annotations

from typing import Any

import msgspec

# ------------------------------ control messages ------------------------------
# Dispatched by reserved topic prefix (acies/ctrl/). Consumed by the router;
# never forwarded to user handlers.


class AciesHeartbeat(msgspec.Struct, frozen=True):
    state: str


class AciesGet(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    keys: list[str]


class AciesSet(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    items: dict[str, Any]


class AciesDelete(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    keys: list[str]


class AciesRoute(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    old_topic: str | None
    new_topic: str | None


# ------------------------------- data messages -------------------------------
# Forwarded to user handlers. Users may also define their own msgspec.Struct
# types for application-specific payloads.


class AciesTensor(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    payload: list[float | int | bytes]
    metadata: dict | None = None
