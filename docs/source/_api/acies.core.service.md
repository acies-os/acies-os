# acies.core.service

### Functions

| [`get_cpu_avg_load`](#acies.core.service.get_cpu_avg_load)()        | Get the average CPU load.                                     |
|---------------------------------------------------------------------|---------------------------------------------------------------|
| [`get_mem_info`](#acies.core.service.get_mem_info)()                | Get memory information.                                       |
| [`get_sys_info`](#acies.core.service.get_sys_info)()                | Get system information.                                       |
| [`get_zconf`](#acies.core.service.get_zconf)(mode, connect, listen) | Build a Zenoh configuration from optional CLI-like arguments. |

### Classes

| [`Service`](#acies.core.service.Service)(conf, \*args, \*\*kwargs)   | Base class for long-running Acies services over Zenoh.   |
|----------------------------------------------------------------------|----------------------------------------------------------|

### *class* acies.core.service.Service(conf, \*args, \*\*kwargs)

Bases: [`object`](https://docs.python.org/3/library/functions.html#object)

Base class for long-running Acies services over Zenoh.

This class wraps Zenoh session management, pub/sub registration, control-message
routing, a lightweight scheduler for periodic/one-shot tasks, and liveness/
diagnostic heartbeats. Subclasses implement [`run()`](#acies.core.service.Service.run) to register work
(e.g., schedule tasks or consume `msg_q`) and call [`start()`](#acies.core.service.Service.start) to launch
the control thread and enter the scheduler loop.

* **Parameters:**
  * **conf** (*zenoh.Config*) – Configuration used to open the Zenoh session.
  * **namespace** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *|* *None* *,* *optional*) – Namespace prefix applied by
    [`ns_topic_str()`](#acies.core.service.Service.ns_topic_str) when constructing topics.
  * **proc_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Process name used in logs and in the default
    control topic when `ctrl_topic` is omitted. Defaults to the Zenoh ZID.
  * **ctrl_topic** ([*str*](https://docs.python.org/3/library/stdtypes.html#str) *,* *optional*) – Control topic for `get`/`set`/`topic`/`reply`
    messages. Defaults to `ns_topic_str(proc_name, "ctl")`.
  * **topic** (*Iterable* *[*[*str*](https://docs.python.org/3/library/stdtypes.html#str) *]* *,* *optional*) – Additional topics to subscribe to at start.
  * **deactivated** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – If `True`, application messages are dropped.
  * **enable_heartbeat** ([*bool*](https://docs.python.org/3/library/functions.html#bool) *,* *optional*) – If `True`, emit heartbeat messages.
  * **heartbeat_interval_s** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Seconds between heartbeats. Defaults to 1.
  * **diagnostic_interval_s** ([*int*](https://docs.python.org/3/library/functions.html#int) *,* *optional*) – Minimum seconds between diagnostic
    payloads included with heartbeats. Defaults to 3.

#### session

Active Zenoh session.

* **Type:**
  zenoh.Session

#### active_subs

Current subscriptions (thread-safe).

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), zenoh.Subscriber]

#### active_pubs

Lazily created publishers (thread-safe).

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), zenoh.Publisher]

#### msg_q

Queue of application messages
forwarded from the control thread.

* **Type:**
  [queue.Queue](https://docs.python.org/3/library/queue.html#queue.Queue)[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [AciesMsg](acies.core.types.md#acies.core.types.AciesMsg)]]

#### service_states

Mutable service state/parameters. Includes keys like
`"deactivated"` and `"enable_heartbeat"`, and any values set via
control messages.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### namespace

Namespace used by [`ns_topic_str()`](#acies.core.service.Service.ns_topic_str).

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str) | None

#### proc_name

Process identifier used in topics and logs.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### ctrl_topic

Control topic for this service.

* **Type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### heartbeat_interval_s

Heartbeat period in seconds.

* **Type:**
  [int](https://docs.python.org/3/library/functions.html#int)

#### diagnostic_interval_s

Interval gating diagnostic payloads.

* **Type:**
  [datetime.timedelta](https://docs.python.org/3/library/datetime.html#datetime.timedelta)

#### last_diagnostic

Timestamp of the last diagnostic emission.

* **Type:**
  [datetime.datetime](https://docs.python.org/3/library/datetime.html#datetime.datetime)

#### event

Stop signal for background control processing.

* **Type:**
  [threading.Event](https://docs.python.org/3/library/threading.html#threading.Event)

#### \_scheduler

Scheduler for periodic/one-shot tasks.

* **Type:**
  [sched.scheduler](https://docs.python.org/3/library/sched.html#sched.scheduler)

Message flow:
: Incoming Zenoh samples are delivered to an internal queue and processed
  by a background control thread. Control messages of kind `"set"`,
  `"get"`, `"topic"`, and `"reply"` are handled internally; all other
  messages are forwarded to [`msg_q`](#acies.core.service.Service.msg_q) for application logic.
  Use [`make_msg()`](#acies.core.service.Service.make_msg) / [`make_reply()`](#acies.core.service.Service.make_reply) to construct messages and
  [`send()`](#acies.core.service.Service.send) to publish them. Supported kinds include:
  `array_i16`, `array_i32`, `array_i64`, `array_f64`, `json`,
  `heartbeat`, `get`, `set`, `topic`, and `reply`.

Lifecycle:
: * Call [`start()`](#acies.core.service.Service.start) to spawn the control thread, schedule heartbeats,
    invoke [`run()`](#acies.core.service.Service.run), and enter the scheduler loop.
  * Override [`run()`](#acies.core.service.Service.run) in subclasses to register work (e.g., via
    [`schedule()`](#acies.core.service.Service.schedule)) or to consume [`msg_q`](#acies.core.service.Service.msg_q).
  * Override [`shutdown()`](#acies.core.service.Service.shutdown) for application-specific cleanup. Internal
    resources (subs/pubs/session) are released by the framework via
    `_undeclare()`.

Thread-safety:
: Subscription and publisher maps are protected by locks. [`send()`](#acies.core.service.Service.send)
  creates publishers on first use. Use [`remove_sub()`](#acies.core.service.Service.remove_sub) / [`remove_pub()`](#acies.core.service.Service.remove_pub)
  or rely on shutdown to release resources.

#### \_\_init_\_(conf, \*args, \*\*kwargs)

* **Parameters:**
  **conf** (`Config`)

#### add_sub(topic)

Subscribe to a topic if not already subscribed.

Declares a Zenoh subscriber with best-effort reliability and
registers the internal handler.

* **Parameters:**
  **topic** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The topic to subscribe to.

#### *static* decode_payload(data)

Deserialize payload bytes to a Python object.

Tries pickle first, then JSON (UTF-8). Raises on failure.

* **Parameters:**
  **data** – Raw payload bytes.
* **Returns:**
  Decoded Python object.
* **Return type:**
  Any
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If the data cannot be decoded as pickle or JSON.

#### *static* encode_payload(data)

Serialize a payload to bytes for transport.

Uses JSON (UTF-8) by default; raise `ValueError` if not serializable.

* **Parameters:**
  **data** – JSON-serializable Python object.
* **Returns:**
  Serialized payload.
* **Return type:**
  [bytes](https://docs.python.org/3/library/stdtypes.html#bytes)
* **Raises:**
  [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If `data` cannot be JSON-encoded.

#### get_zid()

Return the Zenoh session ZID as a string

* **Returns:**
  The ZID as a string.
* **Return type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

#### make_msg(kind, payload, meta=None, reply_to=None, timestamp_ns=None)

Construct an `AciesMsg` of a given kind.

Supported kinds:
: - `"array_<dtype>"`: payload must be `list`; `dtype` in {`i16`, `i32`, `i64`, `f64`}.
  - `"json"`: payload must be `dict`.
  - `"heartbeat"`: no payload requirements.
  - Control kinds: `"get"`, `"set"`, `"topic"`, `"reply"`.

* **Parameters:**
  * **kind** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – Message kind selector (see above).
  * **payload** ([`Any`](https://docs.python.org/3/library/typing.html#typing.Any)) – Message payload (type depends on `kind`).
  * **meta** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/library/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]) – Optional metadata dict to attach.
  * **reply_to** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`str`](https://docs.python.org/3/library/stdtypes.html#str)]) – Topic to address; defaults to [`ctrl_topic`](#acies.core.service.Service.ctrl_topic) if `None`.
  * **timestamp_ns** ([`int`](https://docs.python.org/3/library/functions.html#int) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional nanosecond timestamp for the message.
* **Returns:**
  The constructed message.
* **Return type:**
  [AciesMsg](acies.core.types.md#acies.core.types.AciesMsg)
* **Raises:**
  * [**AssertionError**](https://docs.python.org/3/library/exceptions.html#AssertionError) – If payload type does not match the selected `kind`.
  * [**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError) – If `kind` is not one of the supported kinds.

#### make_reply(req_msg, reply_payload, timestamp_ns=None)

Create a reply control message for a prior request.

* **Parameters:**
  * **req_msg** ([`AciesMsg`](acies.core.types.md#acies.core.types.AciesMsg)) – The original request message being answered.
  * **reply_payload** ([`dict`](https://docs.python.org/3/library/stdtypes.html#dict)) – Arbitrary reply payload to embed under `"respond"`.
  * **timestamp_ns** ([`int`](https://docs.python.org/3/library/functions.html#int) | [`None`](https://docs.python.org/3/library/constants.html#None)) – Optional nanosecond timestamp to stamp on the message.
* **Returns:**
  A control message of kind `"reply"` addressed to
  [`ctrl_topic`](#acies.core.service.Service.ctrl_topic) containing the response and request metadata.
* **Return type:**
  [AciesMsg](acies.core.types.md#acies.core.types.AciesMsg)

#### ns_topic_str(\*args)

* **Return type:**
  [`str`](https://docs.python.org/3/library/stdtypes.html#str)

#### remove_pub(topic)

Undeclare and remove a cached publisher for a topic.

* **Parameters:**
  **topic** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – Topic whose publisher should be removed.

#### remove_sub(topic)

Remove a subscription to a topic.

* **Parameters:**
  **topic** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The topic to unsubscribe from.

#### run()

#### schedule(delay, func, priority=1, periodic=False, absolute_time=False, func_args=())

Schedule a function to run after a delay. If periodic is True, the function will be called repeatedly
:type delay: [`float`](https://docs.python.org/3/library/functions.html#float)
:param delay: Seconds until the call (or absolute timestamp if `absolute_time=True`).
:type func: [`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)
:param func: Callable to execute.
:type priority: [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
:param priority: Lower values run first if multiple events share the same time.
:type periodic: [`bool`](https://docs.python.org/3/library/functions.html#bool)
:param periodic: If True, reschedules itself with the same delay after each run.
:type absolute_time: [`bool`](https://docs.python.org/3/library/functions.html#bool)
:param absolute_time: If True, `delay` is treated as an absolute timestamp.
:type func_args: [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple)
:param func_args: Positional arguments to pass to `func`.

* **Returns:**
  The scheduled event (can be canceled).
* **Return type:**
  sched.Event

#### send(topic, msg)

Send a message to a specific topic.

* **Parameters:**
  * **topic** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – The topic to send the message to.
  * **msg** ([*AciesMsg*](acies.core.types.md#acies.core.types.AciesMsg)) – The message to send.

#### *property* service_states

Mutable service state dictionary.

Holds configuration/state values (e.g., `"deactivated"`,
`"enable_heartbeat"`) and any keys set via control messages.

* **Type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)

#### shutdown()

#### start()

Start the service control thread, schedule heartbeats, and enter the scheduler loop.

Spawns the control-message thread, sets up periodic heartbeat emission,
invokes [`run()`](#acies.core.service.Service.run), and then runs the scheduler until interrupted.
Ensures shutdown and resource cleanup in a `finally` block.

#### topic_str(\*args)

Build a topic string by joining components with `'/'`.

`None` components are skipped.

* **Parameters:**
  **\*args** – Topic path segments.
* **Returns:**
  The joined topic string.
* **Return type:**
  [str](https://docs.python.org/3/library/stdtypes.html#str)

### acies.core.service.get_cpu_avg_load()

Get the average CPU load.

* **Returns:**
  A dictionary containing the CPU load averages for 1, 5, and 15 minutes.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]

### acies.core.service.get_mem_info()

Get memory information.

* **Returns:**
  A dictionary containing memory information.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]

### acies.core.service.get_sys_info()

Get system information.

* **Returns:**
  A dictionary containing CPU and memory information.
* **Return type:**
  [dict](https://docs.python.org/3/library/stdtypes.html#dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [float](https://docs.python.org/3/library/functions.html#float)]

### acies.core.service.get_zconf(mode, connect, listen)

Build a Zenoh configuration from optional CLI-like arguments.

* **Parameters:**
  * **mode** – Zenoh mode value, or None.
  * **connect** – Endpoint(s) to connect to, or None.
  * **listen** – Endpoint(s) to listen on, or None.
* **Returns:**
  Populated configuration object.
* **Return type:**
  zenoh.Config
