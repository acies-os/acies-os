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

Triggered by an incoming message on one or more topics.

```python
@app.subscribe('sensors/temp', 'sensors/humidity')
def handle(ctx: AciesContext, msg: AciesMsg):
    ctx.publish('processed/temp', {'value': msg.payload['value'] * 1.8 + 32})
```

### SCHEDULE — timer-driven

Triggered at a fixed interval. No message is delivered.

```python
@app.schedule(interval=1.0)
def poll(ctx: AciesContext):
    ctx.publish('sensors/temp', {'value': read_hw_sensor()})
```

### SERVICE — RPC / queryable

Triggered by an incoming query. The handler's return value is sent as the
reply. Implemented using zenoh's query/queryable mechanism.

```python
@app.service('rpc/status')
def status(ctx: AciesContext, msg: AciesMsg) -> dict:
    return {'node': app.name, 'ok': True}
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
            ctx.publish('sensors/audio', {'data': raw})
    threading.Thread(target=mic_thread, daemon=True).start()
```

No special decorator is needed. The framework provides `ctx`; the user owns
the thread.

## Handler Calling Convention

`ctx` is constructed once at app startup and passed to every handler. `msg` is
constructed by the router when it creates a `Job` for an incoming message.

Dispatch rule:

- SUBSCRIBE / SERVICE: `fn(ctx, msg)`
- SCHEDULE: `fn(ctx)` — no message

Determined at runtime by whether `job.msg is None`. No per-call inspection or
injection machinery is needed.

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

    def publish(self, topic: str, payload: Any, metadata: dict | None = None) -> None: ...
    def query(self, topic: str, payload: Any = None, timeout: float = 1.0) -> AciesMsg | None: ...
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
@app.subscribe('sensors/temp')
def accumulate(ctx: AciesContext, msg: AciesMsg):
    # task-local: only this handler's jobs touch this state
    with ctx.task.lock:
        ctx.task.data.setdefault('readings', []).append(msg.payload['value'])

@app.schedule(interval=5.0)
def report(ctx: AciesContext):
    # cross-handler: share a value produced by another handler
    with ctx.app.lock:
        last = ctx.app.data.get('last_temp')
    if last is not None:
        ctx.publish('reports/temp', {'last': last})
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

- **Router** — inbound queue + router thread. Transports push `(topic, msg)`
  into the queue. The router thread matches topics to `TaskSpec`s, creates
  `Job`s, and calls `executor.enqueue()`. Outbound (`publish`, `query`) is
  synchronous, called directly from worker threads via `ctx`.

- **Executor** — internal FIFO queue + dispatcher thread + worker thread pool.
  Dequeues `Job`s and submits them to the pool. After each handler returns,
  calls `job.reply_fn(result)` if set (SERVICE jobs only).

- **Timer thread** — one shared thread for all SCHEDULE specs. Uses a priority
  queue of `(next_fire_time, spec)` to enqueue jobs at the right time.

- **Transport layer** — indirection over the messaging backend, defined as a
  `Protocol` (structural typing, no inheritance required). Three backends:
  `ZenohTransport` (production pub/sub; same-host IPC via Zenoh UDS/SHM config),
  `WebSocketTransport` (browser/UI; own server thread, `ws://` topic prefix),
  `LocalTransport` (in-process queue, for tests).
