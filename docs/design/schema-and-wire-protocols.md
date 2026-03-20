# Schema and Wire Protocol Design

AciesOS components communicate over Eclipse Zenoh. This document covers how
messages are defined (schema) and encoded on the wire. These are independent
concerns.

---

## Schema

### Decision: `msgspec.Struct`

Message types are defined as `msgspec.Struct` subclasses — pure data
containers with no methods. Validation happens at decode time (the boundary),
not inside components.

```python
class SensorReading(msgspec.Struct, frozen=True):
    value: float
    timestamp: int
```

| Option                    | Notes                                                               |
| ------------------------- | ------------------------------------------------------------------- |
| Plain dicts / lists       | Flexible. Silent failures at boundaries. No IDE support.            |
| `TypedDict` / `dataclass` | Python-native. Require a separate serialization step.               |
| **`msgspec.Struct`**      | Pure data, validated at decode time, fast.                          |
| Protobuf / Avro           | Strong schema enforcement. Requires codegen and separate toolchain. |

### On `AciesMsg` as a base class

We considered `class AciesMsg(msgspec.Struct): pass` to hide `msgspec` from
users. Rejected because:

- `msgspec` is committed as an explicit dependency — there is nothing to hide.
- Users still need to understand msgspec concepts (`frozen=True`, field
  definitions). A thin wrapper signals insulation that does not exist.
- Cross-language interop is a wire-level concern; hiding the Python library
  provides no benefit there.

`msgspec` is a first-class dependency. Users import and use it directly.

---

## Wire Protocol

### Decision: MessagePack via `msgspec`

MessagePack is the default wire format via `msgspec.msgpack`.

When `msgspec` encodes a `Struct`, it uses **array encoding** — fields are
serialized positionally with no field names on the wire:

```python
# SensorReading(value=22.5, timestamp=1234567890)
# Wire: [22.5, 1234567890]  — values only, no keys
```

This eliminates key-repetition overhead and produces sizes comparable to
Protobuf, without any IDL or codegen.

| Format                  | Size      | Speed         | Binary-native | Tooling              |
| ----------------------- | --------- | ------------- | ------------- | -------------------- |
| JSON                    | Large     | Slow          | No (base64)   | None                 |
| MessagePack (map)       | Medium    | Fast          | Yes           | None                 |
| **MessagePack (array)** | **Small** | **Very fast** | **Yes**       | **None**             |
| Protobuf                | Small     | Very fast     | Yes           | High (IDL + codegen) |
| FlatBuffers             | Smallest  | Fastest       | Yes           | High (IDL + codegen) |

Protobuf and FlatBuffers offer marginal size gains over array-encoded
MessagePack. For binary-heavy workloads (audio, tensors), the dominant cost is
the raw data, not the framing. The toolchain complexity is not justified.

### Cross-language interoperability

Interop is governed by the wire format, not the Python library. Any language
with a MessagePack implementation can participate in AciesOS topics. A Rust
component uses `rmp-serde`; C++ uses `msgpack-cxx`. They interoperate through
the agreed wire schema.

JSON encoding (`msgspec.json`) is available as an opt-in for debugging with
standard Zenoh CLI tools.

---

## Summary

| Concern                | Decision                                                  |
| ---------------------- | --------------------------------------------------------- |
| Message schema         | `msgspec.Struct` — pure data, validated at boundaries     |
| Wire format            | MessagePack (array encoding via `msgspec.msgpack`)        |
| Abstraction layer      | None — `msgspec` is an explicit, first-class dependency   |
| Cross-language interop | Governed by wire schema + MessagePack, not Python library |
