# acies.core.types

### Classes

| [`AciesMsg`](#acies.core.types.AciesMsg)()   | High-level wrapper around the C-extension `_Msg`.   |
|----------------------------------------------|-----------------------------------------------------|

### *class* acies.core.types.AciesMsg

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

High-level wrapper around the C-extension `_Msg`.

Provides constructors for array, json, heartbeat, and control messages,
plus helpers to access payload/metadata as Python objects and to
convert to/from raw bytes.

### Notes

* Timestamps are nanoseconds since epoch.
* Payloads/metadata are stored as UTF-8 JSON when applicable.

#### \_\_init_\_()

#### *classmethod* from_bytes(data)

Create a message from its wire-format bytes.

* **Parameters:**
  **data** ([`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes)) – Byte representation produced by [`to_bytes()`](#acies.core.types.AciesMsg.to_bytes).
* **Returns:**
  Decoded message instance.
* **Return type:**
  [AciesMsg](#acies.core.types.AciesMsg)

#### get_metadata()

Return the decoded metadata.

* **Returns:**
  Decoded metadata mapping (empty dict if `None`).
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### get_payload()

Return the decoded payload.

* **Returns:**
  Decoded payload mapping (empty dict if `None`).
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### *property* kind *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

Message kind (e.g., `"json"`, `"reply"`, `"array_i16"`).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### *classmethod* new_array_msg(payload, reply_to, metadata, data_type='i16', timestamp_ns=None)

Create an array message.

* **Parameters:**
  * **payload** ([`list`](https://docs.python.org/3/library/stdtypes.html#list)[[`int`](https://docs.python.org/3/library/functions.html#int) | [`float`](https://docs.python.org/3/library/functions.html#float)]) – Numeric values to send.
  * **reply_to** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Topic to address replies to.
  * **metadata** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional mapping attached as metadata.
  * **data_type** – One of `{"i16","i32","i64","f64"}`.
  * **timestamp_ns** ([`int`](https://docs.python.org/3/library/functions.html#int) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional nanosecond timestamp to set.
* **Returns:**
  The constructed message.
* **Return type:**
  [AciesMsg](#acies.core.types.AciesMsg)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If `data_type` is not supported.

#### *classmethod* new_ctl_msg(kind, reply_to, payload, metadata=None, timestamp_ns=None)

Create a control message.

* **Parameters:**
  * **kind** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – One of `{"set","get","topic","reply"}`.
  * **reply_to** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Control topic.
  * **payload** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – Control payload mapping (JSON-serializable).
  * **metadata** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional metadata mapping.
  * **timestamp_ns** ([`int`](https://docs.python.org/3/library/functions.html#int) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional nanosecond timestamp to set.
* **Returns:**
  The control message.
* **Return type:**
  [AciesMsg](#acies.core.types.AciesMsg)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If `kind` is not supported.

#### *classmethod* new_heartbeat(reply_to, metadata, timestamp_ns=None)

Create a heartbeat message.

* **Parameters:**
  * **reply_to** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Topic to address replies to.
  * **metadata** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional mapping attached as metadata.
  * **timestamp_ns** ([`int`](https://docs.python.org/3/library/functions.html#int) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional nanosecond timestamp to set.
* **Returns:**
  The heartbeat message.
* **Return type:**
  [AciesMsg](#acies.core.types.AciesMsg)

#### *classmethod* new_json_msg(reply_to, payload, metadata=None, timestamp_ns=None)

Create a JSON message.

* **Parameters:**
  * **reply_to** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Topic to address replies to.
  * **payload** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – JSON-serializable mapping as the message body.
  * **metadata** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional metadata mapping.
  * **timestamp_ns** ([`int`](https://docs.python.org/3/library/functions.html#int) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional nanosecond timestamp to set.
* **Returns:**
  The JSON message.
* **Return type:**
  [AciesMsg](#acies.core.types.AciesMsg)

#### *property* reply_to *: [str](https://docs.python.org/3/library/stdtypes.html#str)*

#### set_metadata(value)

Set metadata for message kinds that support it.

Supported kinds: `array_i16`, `array_i32`, `array_i64`,
`array_f64`, `set`, `get`, `reply`, `topic`.

* **Parameters:**
  **value** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – Mapping to serialize as metadata.
* **Raises:**
  [**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError) – If the current message kind does not support metadata.

#### *property* timestamp *: [int](https://docs.python.org/3/library/functions.html#int)*

#### to_bytes()

Serialize the message to its wire format.

* **Returns:**
  Encoded message suitable for transport/storage.
* **Return type:**
  [bytes](https://docs.python.org/3/library/stdtypes.html#bytes)

#### to_dict()

Return a JSON-friendly dictionary view of the message.

* **Returns:**
  Mapping with keys `"kind"`, `"timestamp"`, `"reply_to"`,
  `"payload"`, and `"metadata"`.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### to_json()

Return a JSON string representation of [`to_dict()`](#acies.core.types.AciesMsg.to_dict).

* **Returns:**
  JSON-encoded message summary.
* **Return type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)
