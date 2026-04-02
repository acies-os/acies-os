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


class AciesKvChange(msgspec.Struct, frozen=True):
    """Published to ctl/notify/<key> after a successful kv set or del."""

    key: list[str]
    op: str  # 'set' | 'del'
    value: Any = None  # new value for 'set'; None for 'del'


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


class AciesTimeSeries(msgspec.Struct, frozen=True):
    """Time-series sensor message with typed, numpy-compatible payload.

    ``payload[i]`` contains all samples for ``channels[i]`` as raw bytes.
    Use ``np.frombuffer(msg.payload[i], dtype=msg.dtype)`` to decode.

    Example (single-channel geo, 200 Hz, 1-second window)::

        AciesTimeSeries(
            source='edge-01/geo',
            timestamp=...,
            payload=[np.array(samples, dtype=np.int32).tobytes()],
            channels=['SH3'],
            sampling_rate=200,
            dtype='int32',
        )
    """

    source: str
    timestamp: NanoSecond
    payload: list[bytes]  # payload[i] = raw samples for channels[i]
    channels: list[int] | list[str]
    sampling_rate: int
    dtype: str  # numpy dtype string: 'int16', 'int32', etc.


class AciesPrediction(msgspec.Struct, frozen=True, omit_defaults=True):
    """One ranked prediction from a classifier.

    ``label`` and ``score`` are always present. Optional fields are omitted
    from the wire encoding when absent (``omit_defaults=True``).

    Example (truck detection with range and location)::

        AciesPrediction(
            label='truck',
            score=0.92,
            distance=340.5,
            latitude=37.7749,
            longitude=-122.4194,
        )
    """

    label: str
    score: float
    distance: float | None = None  # estimated range in metres
    speed: float | None = None  # estimated speed in m/s
    latitude: float | None = None  # decimal degrees
    longitude: float | None = None  # decimal degrees
    extras: dict[str, Any] = {}  # model-specific metadata

    def __repr__(self) -> str:
        parts = [f'label={self.label!r}', f'score={self.score:.3f}']
        if self.distance is not None:
            parts.append(f'distance={self.distance:.3f}')
        if self.speed is not None:
            parts.append(f'speed={self.speed:.3f}')
        if self.latitude is not None:
            parts.append(f'lat={self.latitude:.3f}')
        if self.longitude is not None:
            parts.append(f'lon={self.longitude:.3f}')
        if self.extras:
            parts.append(f'extras={self.extras!r}')
        return f'Pred({", ".join(parts)})'


class AciesInference(msgspec.Struct, frozen=True):
    """Inference result message published by a classifier.

    Example (two-target vehicle classification)::

        AciesInference(
            source='edge-01/classifier',
            timestamp=...,
            predictions=[
                AciesPrediction(label='truck', score=0.92),
                AciesPrediction(label='car', score=0.61, distance=340.5),
            ],
        )
    """

    source: str
    timestamp: NanoSecond
    predictions: list[AciesPrediction]
