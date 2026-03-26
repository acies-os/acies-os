# Acies-OS 2.0 Design Document

## Overview

Acies-OS is a middleware designed for Edge AI applications. It assumes a
computation graph model where nodes represent computational tasks and edges
represent data dependencies between these tasks. The communication layer is
based on a pub-sub system.

The refactor of the 2.0 version is to:

1. Upgrade the pubsub layer to the latest version (zenoh >= 1.0)
2. Provide a decorator-based API for users to define their computation graph
   (akin to Ray, Flask, FastAPI, etc.)
3. Provide internal abstractions for future research and improvements, such as
   different scheduling policies.

## Task Kinds

The user code creates an instance of `AciesApp`, then defines handler functions
decorated with one of three decorators. Decorators describe the **trigger
mechanism** — what causes a task to run. Whether a handler reads input only
(sink) or also publishes output (transform) is determined by whether it calls
`ctx.publish()`, not by the decorator.

Each kind is a distinct frozen dataclass (`SubscriberSpec`, `ScheduleSpec`,
`ServiceSpec`). Their union is the `TaskSpec` type. This sum type pattern makes
invalid states unrepresentable — a `ScheduleSpec` cannot have topics; a
`SubscriberSpec` cannot have an interval.

### SUBSCRIBE — message-driven

Triggered by an incoming message on one or more topics. The `msg` parameter is
annotated with a `msgspec.Struct` type; the router uses it to decode the wire
bytes. Users define their own types or use built-in types like `AciesTensor`.

```python
import msgspec
from acies.corev2 import Topic

class SensorReading(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    value: float

@app.subscribe(Topic('sensor/temp'), Topic('sensor/humidity'))
def handle(ctx: AciesContext, msg: SensorReading):
    # Inside handlers, use ctx.ns to construct topics as plain strings
    ctx.publish(ctx.ns.topic('processor/temp'), SensorReading(
        source='edge-01', timestamp=msg.timestamp, value=msg.value * 1.8 + 32
    ))
```

### SCHEDULE — timer-driven

Triggered at a fixed interval. No message is delivered.

```python
@app.schedule(interval=1.0)
def poll(ctx: AciesContext):
    ctx.publish(ctx.ns.topic('sensor/temp'), SensorReading(
        source='edge-01', timestamp=now_ns(), value=read_hw_sensor()
    ))
```

### SERVICE — RPC / queryable

Triggered by an incoming query. The handler's return value is sent as the
reply. Implemented using zenoh's query/queryable mechanism.

```python
from acies.corev2 import CtlTopic

class StatusReply(msgspec.Struct, frozen=True):
    source: str
    ok: bool

@app.service(CtlTopic('status'))
def status(ctx: AciesContext, msg: msgspec.Struct) -> StatusReply:
    return StatusReply(source='edge-01', ok=True)
```

## Topic Namespace

Topics are free-form zenoh key expressions. The framework imposes no mandatory
structure on data topics — applications may organise them domain-centrically
(ROS/MQTT style) or device-centrically, or mix both.

Control topics are the only exception: they are always addressed to a specific
device/service instance and follow `<host>/<name>/ctl/<service>`.

### Topic types

Three forms are accepted at decoration time (`app.subscribe`, `app.service`):

| Form | Example | Resolves to |
|------|---------|-------------|
| `str` | `'building/a/temperature'` | used as-is |
| `str` with `{key}` | `'{input_topic}'` or `'section/{section}/room/{room}'` | resolved from `app.state.config` via `format_map` at `run()` |
| `Topic(path, prefix=True)` | `Topic('audio/raw')` | `edge-01/mic/audio/raw` |
| `CtlTopic(path)` | `CtlTopic('kv')` | `edge-01/mic/ctl/kv` |

`Topic`, `CtlTopic`, and format-string topics are all resolved lazily at
`run()` time, so CLI-overridden `host`/`name` and config values are always
reflected correctly.

`ctx.publish()` and `ctx.query()` accept plain `str` only. For
CLI-configurable output topics, read from `ctx.app.config` directly:

```python
ctx.publish(ctx.app.config['output_topic'], msg)
# or composite:
ctx.publish(f"section/{ctx.app.config['section']}/room/{ctx.app.config['room']}/data", msg)
```

`Topic` prefix options:
- `prefix=True` (default) — prepends `<host>/<name>`
- `prefix=''` or `False` — no prefix (domain-centric)
- `prefix='org/site'` — custom prefix string (verbatim)

### Wildcards

Zenoh key expression wildcards apply to the full topic path:

```python
# All temperature from any device/service
@app.subscribe('**/temperature')

# All outputs from this app
@app.subscribe(Topic('**'))                  # -> edge-01/mic/**

# All ctl heartbeats across all devices
@app.subscribe('**/ctl/heartbeat')

# Domain-centric: all room temperature sensors
@app.subscribe('building/*/room/*/temperature')
```

Wildcard rules: `*` matches one non-empty, non-`/` segment; `**` matches any
number of segments; `$*` is an infix pattern within a segment (e.g.
`thermo$*`). Selector characters `?` and `#` are forbidden.

### Transport selection

The router uses a simple default-plus-prefix-override rule:

| Topic      | Transport used                                                            |
| ---------- | ------------------------------------------------------------------------- |
| `ws://...` | `WebSocketTransport` (browser/UI)                                         |
| everything else | default transport (ZenohTransport in production, LocalTransport in tests) |

Application code never specifies a transport. Moving a service between
processes or hosts requires no code changes.

### Two topic APIs

**Decorator time** — `app.subscribe()` and `app.service()` run at import time,
before CLI arguments are available. Use lazy types that resolve at `run()`:

```python
from acies.corev2 import Topic, CtlTopic

@app.subscribe(Topic('audio/raw'))          # -> edge-01/mic/audio/raw
@app.subscribe(Topic('**'))                 # -> edge-01/mic/**
@app.subscribe(Topic('room/5', prefix=''))  # -> room/5  (domain-centric)
@app.service(CtlTopic('kv'))               # -> edge-01/mic/ctl/kv
@app.subscribe('{input_topic}')            # -> app.state.config['input_topic']
@app.subscribe('section/{section}/room/{room}/temp')  # composite from config
```

**Handler time** — inside handlers, `ctx.ns` is a resolved `Namespace`.
`ctx.publish()` and `ctx.query()` accept plain `str` only:

```python
def handler(ctx: AciesContext, msg: SensorReading):
    ctx.publish(ctx.ns.topic('audio/raw'), AciesTensor(...))  # edge-01/mic/audio/raw
    ctx.publish('building/a/room/5/temp', msg)                # domain-centric
    ctx.publish(ctx.app.config['output_topic'], msg)          # CLI-configurable output
    ctx.query(ctx.ns.ctl.kv, payload, timeout=0.5)            # edge-01/mic/ctl/kv
    ctx.query('edge-02/classifier/ctl/infer', payload, timeout=1.0)  # cross-device
```

## Source Nodes

Some nodes produce data from external sources (audio drivers, cameras, network
streams) rather than reacting to messages or timers. These are plain threads
started in an `on_startup` hook, where `ctx` is already live:

```python
@app.on_startup
def start_mic(ctx: AciesContext):
    def mic_thread():
        while True:
            raw = audio_driver.read()
            ctx.publish('edge-01/mic/audio/raw', AciesTensor(...))
    threading.Thread(target=mic_thread, daemon=True).start()
```

No special decorator is needed. The framework provides `ctx`; the user owns
the thread.

## Handler Calling Convention

`ctx` is constructed once at app startup and passed to every handler. `msg` is
decoded and constructed by the router when it creates a `Job` for an incoming
message, using the type annotation on the handler's `msg` parameter.

Dispatch rule:

- SUBSCRIBE / SERVICE: `fn(ctx, msg)`
- SCHEDULE: `fn(ctx)` — no message

Determined at runtime by whether `job.raw is None`. No per-call inspection or
injection machinery is needed.

The `msg` type annotation is extracted at decoration time and stored as
`spec.msg_type`. The router calls `msgspec.msgpack.decode(raw, type=spec.msg_type)`
on each inbound message. If no annotation is provided, falls back to
`msgspec.msgpack.decode(raw)` (plain dict/list).

For `ServiceSpec`, the return type annotation is also extracted and stored as
`spec.return_type`. It is used at startup to generate a JSON Schema for the
response (see `sys.schemas` below). It has no effect on dispatch.

> **Future extension**: `Depends(fn)` markers in the handler signature (akin
> to FastAPI) are a planned extension for injecting app-level resources
> (database sessions, model handles, etc.). Out of scope for Phase 2.

## AciesContext

Passed to every handler and lifecycle hook. Provides pub/sub access and two
state stores for stateful computation.

```python
class AciesContext:
    app: AppState   # global store — shared across all handlers in this app
    task: TaskState # task store  — shared across all jobs of this handler only

    def publish(self, topic: str, msg: msgspec.Struct) -> None: ...
    def query(self, topic: str, msg: msgspec.Struct, timeout: float = 1.0) -> msgspec.Struct | None: ...
```

Handlers never touch the transport directly.

One `AciesContext` instance is constructed per `TaskSpec` at `run()` time and
reused across all jobs of that spec — no per-job allocation.

## AppState and TaskState

`AppState` has two dicts and a lock. Thread safety is the caller's
responsibility: acquire the lock for any compound read-modify-write operation.

```python
class AppState:
    lock: threading.RLock
    config: dict   # externally controllable; 'sys' key reserved for middleware
    data: dict     # free-form transient state; internal to the app

class TaskState:
    lock: threading.RLock
    data: dict
```

`config` is populated from CLI arguments (via `app.cli()`) and may be updated
externally via AciesSet (planned). The `sys` key is reserved:

```python
app.state.config['sys']  # {'host': ..., 'name': ..., 'state': ..., 'schemas': {...}}
```

`sys.schemas` is populated at `run()` time from all registered `ServiceSpec`s.
Each entry holds the resolved topic string, a `request` JSON Schema (from
`msgspec.json.schema(spec.msg_type)`), and a `response` JSON Schema (from
`spec.return_type`). Entries with untyped handlers omit the corresponding key.
`sys.schemas` is readable via `ctl/kv` but protected from mutation.

`data` is free-form transient state not externally controlled. Usage:

```python
@app.subscribe('**/sensor/temp')
def accumulate(ctx: AciesContext, msg: SensorReading):
    with ctx.task.lock:
        ctx.task.data.setdefault('readings', []).append(msg.value)

@app.schedule(interval=5.0)
def report(ctx: AciesContext):
    with ctx.app.lock:
        last = ctx.app.data.get('last_temp')
    if last is not None:
        ctx.publish(ctx.ns.topic('aggregator/reports/temp'), SensorReading(...))
```

## Control Plane

Every `AciesApp` automatically registers four built-in services under
`<host>/<name>/ctl/`:

| Topic          | Request type          | Purpose                                              |
|----------------|-----------------------|------------------------------------------------------|
| `ctl/heartbeat`| —                     | Periodic pub of `AciesHeartbeat`; not a service      |
| `ctl/kv`       | `AciesKvRequest`      | Get / set / del on `app.state.config`                |
| `ctl/route`    | `AciesRouteRequest`   | Reroute or suppress input/output topics at runtime   |
| `ctl/io`       | `AciesIoRequest`      | Snapshot of task-to-topic I/O routing table          |
| `ctl/schema`   | `AciesSchemaRequest`  | JSON Schema for each registered service's types      |

**`ctl/kv`** — `sys` key is partially protected: `sys.state` is the only mutable
sub-key; all other `sys.*` entries and `sys` itself are read-only.

**`ctl/route`** — accepts a list of `TopicRename` for inputs (subscribe/unsubscribe
at runtime) and outputs (redirect or suppress publishes). Consecutive renames
collapse: `t1->t2` then `t2->t3` results in a single effective entry `t1->t3`.

**`ctl/io`** — returns `{spec_id: {name, inputs: [...], outputs: [...]}}`.
Inputs are the topics registered at startup; outputs accumulate as the handler
publishes at runtime (conditional paths appear only after they are exercised).

**`ctl/schema`** — returns `{spec_id: {name, topic, request?, response?}}` for
every registered `ServiceSpec`. Schema values are JSON Schema dicts produced by
`msgspec.json.schema()`. The same data is stored in `sys.schemas` and is
readable via `ctl/kv get ['sys', 'schemas']`.

## Lifecycle Hooks

`on_startup` runs after all subsystems are started, before the app blocks.
`on_shutdown` runs after `stop()` is called, before the app exits. Both receive
`ctx`.

```python
@app.on_startup
def setup(ctx: AciesContext): ...

@app.on_shutdown
def teardown(ctx: AciesContext): ...
```

## Internal Architecture

`AciesApp` owns:

- **Router** — inbound queue + router thread. Transports push
  `(topic, raw_bytes, reply_fn | None)` into the queue; `reply_fn` is `None`
  for pub messages and set (by the transport) for incoming queries. The router
  thread matches topics to `TaskSpec`s and creates `Job`s — it does not encode
  or decode. Outbound (`publish`, `query`) forwards raw bytes directly to the
  transport; `AciesContext` handles encoding before calling the router.

- **Executor** — internal FIFO queue + dispatcher thread + worker thread pool.
  Dequeues `Job`s and calls `dispatch(job)`. The dispatch function (provided
  by `AciesApp`) decodes `job.raw`, calls the handler, and — for SERVICE jobs
  — encodes the result and calls `job.reply_fn(encoded_bytes)` to deliver the
  reply.

- **Timer thread** — one shared thread for all SCHEDULE specs. Uses a priority
  queue of `(next_fire_time, spec)` to enqueue jobs at the right time.

- **Transport layer** — indirection over the messaging backend, defined as a
  `Protocol` (structural typing, no inheritance required). Two backends in
  practice: `ZenohTransport` (production; one session per process, connected
  via unix domain socket to the local zenohd — locality is handled by the
  zenoh network, not the application) and `LocalTransport` (in-process queue;
  test-only, replaces ZenohTransport when no zenoh daemon is available).
  `WebSocketTransport` handles browser/UI connections via the `ws://` prefix.
  The router selects transport by prefix rule only — no topology inference.

## Zenoh Network Topology

Each device runs a local `zenohd` daemon. All processes on the device connect
to it via a unix domain socket (`unix-stream://`). The local daemons are
peered to a server-side `zenohd` that bridges all devices together.

```
Device A                          Device B
┌─────────────────────────┐       ┌─────────────────────────┐
│ [mic]──┐                │       │ [classifier]──┐         │
│        ├──UDS── zenohd ─┼─net───┼─ zenohd ──────┤         │
│ [geo]──┘                │       │ [controller]──┘         │
└─────────────────────────┘       └─────────────────────────┘
                  └──────── server zenohd ────────┘
                       (optional, for bridging)
```

**Locality is handled by the infrastructure:**

- A message from `mic` to `classifier` on the same device travels
  mic -> UDS -> local zenohd -> UDS -> classifier. No network traversal.
- A message to a subscriber on Device B goes through the server zenohd.
- The application code and the router are identical in both cases — one
  `ZenohTransport` session per process, always connecting via UDS to the
  local zenohd.

**`LocalTransport` is test-only.** It replaces `ZenohTransport` in unit and
integration tests so no zenoh daemon is required. In production every process
uses `ZenohTransport`.

### Transport registration

Production:

```python
router.add_transport(ZenohTransport('unix-stream:///run/zenoh/local.sock'))
router.add_transport(WebSocketTransport(), prefix='ws://')
```

Tests:

```python
router.add_transport(LocalTransport())   # replaces ZenohTransport; no daemon needed
```
