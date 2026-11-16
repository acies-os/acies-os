# Acies Wire Protocol

## Generate examples

```python
# %%
from acies.core.types import AciesMsg

# %%
for t in ['i16', 'i32', 'i64', 'f64']:
    m = AciesMsg.new_array_msg([1, 2, 3, 4], 'rs10/geo/ctl', {'key': 'val'}, data_type=t)
    print(m.to_dict())

# %%
m = AciesMsg.new_heartbeat('rs10/geo/ctl', {'key': 'val'})
print(m.to_dict())

# %%
for t in ['set', 'get', 'topic', 'reply']:
    m = AciesMsg.new_ctl_msg(t, 'rs10/geo/ctl', {'key': 'val'}, {'key': 'val'})
    print(m.to_dict())

# %%
m = AciesMsg.new_json_msg('rs10/geo/ctl', {'key': 'val'}, {'key': 'val'})
print(m.to_dict())
```

## Message format

### 1. Array messages

Fields:

- `kind`: `str`, valid value: `'array_i16', 'array_i32', 'array_i64', 'array_f64'`
- `timestamp`: `int`, UNIX epoch in **nanosecond**
- `reply_to`: `str`
- `payload`: `list[int | float]`
- `metadata`: `dict`

Examples:

```
{'kind': 'array_i16', 'timestamp': 1724178136496900000, 'reply_to': 'rs10/geo/ctl', 'payload': [1, 2, 3, 4], 'metadata': {'key': 'val'}}
{'kind': 'array_i32', 'timestamp': 1724178136497017000, 'reply_to': 'rs10/geo/ctl', 'payload': [1, 2, 3, 4], 'metadata': {'key': 'val'}}
{'kind': 'array_i64', 'timestamp': 1724178136497077000, 'reply_to': 'rs10/geo/ctl', 'payload': [1, 2, 3, 4], 'metadata': {'key': 'val'}}
{'kind': 'array_f64', 'timestamp': 1724178136497128000, 'reply_to': 'rs10/geo/ctl', 'payload': [1.0, 2.0, 3.0, 4.0], 'metadata': {'key': 'val'}}
```

### 2. Heartbeat message:

Fields:

- `kind`: `str`, valid value: `heartbeat`
- `timestamp`: `int`, UNIX epoch in **nanosecond**
- `reply_to`: `str`
- `payload`: `dict`, empty
- `metadata`: `dict`

Example:

```
{'kind': 'heartbeat', 'timestamp': 1724178498264494000, 'reply_to': 'rs10/geo/ctl', 'payload': {}, 'metadata': {'key': 'val'}}
```

### 3. Control message:

Fields:

- `kind`: `str`, valid value: `set, get, topic, reply`
- `timestamp`: `int`, UNIX epoch in **nanosecond**
- `reply_to`: `str`
- `payload`: `dict`
- `metadata`: `dict`

Example:

```
{'kind': 'set', 'timestamp': 1724178649881591000, 'reply_to': 'rs10/geo/ctl', 'payload': {'key': 'val'}, 'metadata': {'key': 'val'}}
{'kind': 'get', 'timestamp': 1724178649881715000, 'reply_to': 'rs10/geo/ctl', 'payload': {'key': 'val'}, 'metadata': {'key': 'val'}}
{'kind': 'topic', 'timestamp': 1724178649881777000, 'reply_to': 'rs10/geo/ctl', 'payload': {'key': 'val'}, 'metadata': {'key': 'val'}}
{'kind': 'reply', 'timestamp': 1724178649881833000, 'reply_to': 'rs10/geo/ctl', 'payload': {'key': 'val'}, 'metadata': {'key': 'val'}}
```

### 4. JSON message

Fields:

- `kind`: `str`, valid value: `json`
- `timestamp`: `int`, UNIX epoch in **nanosecond**
- `reply_to`: `str`
- `payload`: `dict`
- `metadata`: `dict`

Example:

```
{'kind': 'json', 'timestamp': 1724178787924859000, 'reply_to': 'rs10/geo/ctl', 'payload': {'key': 'val'}, 'metadata': {'key': 'val'}}
```

## Frontend implementation

Fields:

- `kind`: `str`, valid options:
  - `array_i16`
  - `array_i32`
  - `array_i64`
  - `array_f64`
  - `heartbeat`
  - `set`
  - `get`
  - `topic`
  - `reply`
  - `json`
- `timestamp`: `int`, UNIX epoch in **nanosecond**
- `reply_to`: `str`
- `payload`:
  - if `kind` is one of the array type, then `list[int | float]`
  - else `dict`
- `metadata`: `dict`
