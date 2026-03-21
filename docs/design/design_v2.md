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

class SensorReading(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    value: float

@app.subscribe('*/sensor/temp', '*/sensor/humidity')
def handle(ctx: AciesContext, msg: SensorReading):
    ctx.publish('edge-01/processor/temp', SensorReading(
        source='edge-01', timestamp=msg.timestamp, value=msg.value * 1.8 + 32
    ))
```

### SCHEDULE — timer-driven

Triggered at a fixed interval. No message is delivered.

```python
@app.schedule(interval=1.0)
def poll(ctx: AciesContext):
    ctx.publish('edge-01/sensor/temp', SensorReading(
        source='edge-01', timestamp=now_ns(), value=read_hw_sensor()
    ))
```

### SERVICE — RPC / queryable

Triggered by an incoming query. The handler's return value is sent as the
reply. Implemented using zenoh's query/queryable mechanism.

```python
class StatusReply(msgspec.Struct, frozen=True):
    source: str
    ok: bool

@app.service('edge-01/controller/rpc/status')
def status(ctx: AciesContext, msg: msgspec.Struct) -> StatusReply:
    return StatusReply(source='edge-01', ok=True)
```

## Topic Namespace

All topics follow a three-part hierarchical convention inspired by
Named Data Networking (NDN) / content-centric networking: the name IS the
system. Routing, transport selection, access control, and service discovery
are all derived from the name structure rather than being separate mechanisms.

```
<host>/<service>/<name>
```

| Segment     | Meaning                                   | Example                           |
| ----------- | ----------------------------------------- | --------------------------------- |
| `<host>`    | Unique logical device name                | `edge-01`, `truck-7`              |
| `<service>` | `AciesApp` name                           | `mic`, `classifier`, `controller` |
| `<name>`    | Output name; may contain `/` sub-segments | `audio/raw`, `temp`, `rpc/status` |

`<host>` is assigned at deployment time — by configuration, a UUID combined
with a device label, or a fleet management system. The framework assumes it is
unique within the deployment and stable for the lifetime of the device.

### Transport selection

The router uses a simple default-plus-prefix-override rule — no topology
inference in application code:

| Topic         | Transport used                                                            |
| ------------- | ------------------------------------------------------------------------- |
| `ws://...`    | `WebSocketTransport` (browser/UI)                                         |
| anything else | default transport (ZenohTransport in production, LocalTransport in tests) |

Locality is handled transparently by the zenoh infrastructure (see
[Zenoh Network Topology](#zenoh-network-topology) below), not by the
application or router.

Application code never specifies a transport. Handlers write plain topic
strings. Moving a service between processes or hosts requires no code changes.

### Convention enforcement

All `ctx.publish()` and `ctx.query()` calls in production code must use
fully-qualified topics. The framework validates the `<host>/<service>/`
prefix at publish time and raises if the topic is malformed. The `ws://`
prefix is the only exception (WebSocket topics are browser-facing and do
not follow the three-part convention).

### Cross-device subscriptions and wildcards

Zenoh-style wildcards apply to the full topic path:

```python
# All temperature readings from any device
@app.subscribe('*/sensor/temp')

# All outputs from service 'mic' on any device
@app.subscribe('*/mic/**')

# A specific device's classifier output
@app.subscribe('edge-01/classifier/result')
```

### Example topics

```python
app = AciesApp(name='mic', host='edge-01')

# This service's outputs
ctx.publish('edge-01/mic/audio/raw', AciesTensor(...))
ctx.publish('edge-01/mic/rms', SensorReading(...))

# Calling a service on the same host (routes via zenoh UDS)
ctx.query('edge-01/classifier/rpc/infer', payload, timeout=0.5)

# Calling a service on a remote host (routes via zenoh network)
ctx.query('edge-02/controller/rpc/status', payload, timeout=1.0)
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

Determined at runtime by whether `job.msg is None`. No per-call inspection or
injection machinery is needed.

The `msg` type annotation is extracted at decoration time and stored as
`spec.msg_type`. The router calls `msgspec.msgpack.decode(raw, type=spec.msg_type)`
on each inbound message. If no annotation is provided, falls back to
`msgspec.msgpack.decode(raw)` (plain dict/list).

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

Both are a `lock` + `data` dict. Thread safety is the caller's responsibility:
acquire the lock for any compound read-modify-write operation.

```python
class AppState:
    lock: threading.RLock
    data: dict

class TaskState:
    lock: threading.RLock
    data: dict
```

Usage:

```python
@app.subscribe('*/sensor/temp')
def accumulate(ctx: AciesContext, msg: SensorReading):
    # task-local: only this handler's jobs touch this state
    with ctx.task.lock:
        ctx.task.data.setdefault('readings', []).append(msg.value)

@app.schedule(interval=5.0)
def report(ctx: AciesContext):
    # cross-handler: share a value produced by another handler
    with ctx.app.lock:
        last = ctx.app.data.get('last_temp')
    if last is not None:
        ctx.publish('edge-01/aggregator/reports/temp', SensorReading(...))
```

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

- **Router** — inbound queue + router thread. Transports push `(topic, bytes)`
  into the queue. The router thread checks for control topics (`acies/ctrl/*`,
  handled internally), then matches remaining topics to `TaskSpec`s, decodes
  the bytes using `spec.msg_type`, creates `Job`s, and calls
  `executor.enqueue()`. Outbound (`publish`, `query`) is synchronous, called
  directly from worker threads via `ctx`; the router encodes the struct to
  bytes before handing off to the transport.

- **Executor** — internal FIFO queue + dispatcher thread + worker thread pool.
  Dequeues `Job`s and submits them to the pool. After each handler returns,
  calls `job.reply_fn(result)` if set (SERVICE jobs only).

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
  mic → UDS → local zenohd → UDS → classifier. No network traversal.
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
