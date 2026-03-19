# Implementation Plan v1 for Acies-OS 2.0

## Goals

Implement the v2 middleware described in `design_v2.md` in a new namespace
`acies.corev2`, leaving `acies.core` intact as reference. Once stable, refactor
`acies.controller` and `acies.vehicle_classifier` to use the new API.

---

## 1. Proposed Structure

```
packages/core/src/acies/
├── core/                   # existing — keep untouched as reference
└── corev2/
    ├── __init__.py         # public API surface
    ├── app.py              # AciesApp
    ├── context.py          # AciesContext
    ├── task.py             # SubscriberSpec, ScheduleSpec, ServiceSpec, TaskSpec, Job
    ├── transport.py        # Transport Protocol + LocalTransport (queue-based, for tests)
    ├── router.py           # Router: inbound queue, router thread, topic → Job dispatch
    ├── executor.py         # Executor: internal queue, dispatcher thread, worker pool
    └── msg.py              # AciesMsg: message definition
```

Tests:

```
packages/core/tests/corev2/
    ├── test_task.py
    ├── test_executor.py
    ├── test_router.py
    └── test_app.py         # integration: full app lifecycle with LocalTransport
```

---

## 2. Thread Model

```
Main thread            — orchestration, blocks on stop event
Transport threads      — I/O only; each transport owns its receiver thread(s)
                         and pushes raw messages into the router's inbound queue
Timer thread           — one shared thread for all SCHEDULE specs;
                         calls executor.enqueue() directly
Router thread          — drains router inbound queue; matches topics to TaskSpecs;
                         creates Jobs; calls executor.enqueue()
Executor               — dispatcher thread drains internal queue and submits to
                         worker thread pool
Worker threads         — run handlers; call ctx.publish() synchronously
                         (outbound is direct: worker → router.publish() → transport)
```

---

## 3. Main Entities

### `TaskSpec` (task.py)

Three distinct frozen dataclasses, one per trigger kind. Their union is the
`TaskSpec` type alias. The sum type pattern makes invalid states
unrepresentable: a `ScheduleSpec` cannot have topics; a `SubscriberSpec`
cannot have an interval. No `TaskKind` enum is needed — the type itself is
the discriminant.

```python
@dataclass(frozen=True)
class SubscriberSpec:
    name: str
    fn: Callable
    topics: tuple[str, ...]

@dataclass(frozen=True)
class ScheduleSpec:
    name: str
    fn: Callable
    interval: float

@dataclass(frozen=True)
class ServiceSpec:
    name: str
    fn: Callable
    topics: tuple[str, ...]

TaskSpec = SubscriberSpec | ScheduleSpec | ServiceSpec
```

Source nodes (threads that read from sensors, cameras, etc.) are not a task
kind. They are plain threads started in `on_startup` hooks that call
`ctx.publish()` directly. See `design_v2.md` § Source Nodes.

### `Job` (task.py)

One unit of work, created at runtime from a `TaskSpec`. All three task kinds
go through the executor. `ServiceSpec` jobs carry a `reply_fn` so the executor
can send the reply after the handler returns without the handler knowing about
the underlying query mechanism.

```python
@dataclass
class Job:
    spec: TaskSpec
    msg: AciesMsg | None                      # None for ScheduleSpec jobs
    created_at: float                         # time.monotonic()
    reply_fn: Callable[[Any], None] | None    # only set for ServiceSpec jobs
```

### `AppState` and `TaskState` (context.py)

Two simple state containers: a `threading.RLock` and a `data` dict. Thread
safety is the caller's responsibility — acquire the lock for any compound
read-modify-write.

```python
class AppState:
    lock: threading.RLock
    data: dict

class TaskState:
    lock: threading.RLock
    data: dict
```

### `AciesContext` (context.py)

Passed to every handler and lifecycle hook. One instance is constructed per
`TaskSpec` at `run()` time and reused across all jobs of that spec — no
per-job allocation.

```python
class AciesContext:
    app: AppState   # one instance per AciesApp — shared across all handlers
    task: TaskState # one instance per TaskSpec — shared across jobs of this handler

    def publish(self, topic: str, payload: Any, metadata: dict | None = None) -> None: ...
    def query(self, topic: str, payload: Any = None, timeout: float = 1.0) -> AciesMsg | None: ...
```

> **Future extension**: `Depends(fn)` markers in the handler signature are a
> planned mechanism for injecting per-call resources (DB sessions, model
> handles). No resolver or per-call inspection is needed for Phase 2 — dispatch
> is a simple two-branch callable. See `design_v2.md` § Handler Calling
> Convention.

### `Transport` ABC (transport.py)

Base class for all messaging backends. Each backend owns its receiver thread(s)
and pushes received messages into the router's inbound queue via a callback.
`LocalTransport` (in-process queue) ships alongside the ABC and is used for testing.

```python
class Transport(ABC):
    def start(self, on_message: Callable[[str, AciesMsg], None]) -> None: ...
    def stop(self) -> None: ...
    def can_handle(self, topic: str) -> bool: ...
    def publish(self, topic: str, msg: AciesMsg) -> None: ...
    def subscribe(self, topic: str) -> None: ...
    def query(self, topic: str, msg: AciesMsg, timeout: float) -> AciesMsg | None: ...
    def advertise(self, topic: str, reply_fn_factory: Callable) -> None: ...
```

`start()` receives an `on_message` callback — the transport calls it from its
receiver thread whenever a message arrives, passing `(topic, msg)` to the router.

Concrete backends (extend `Transport`):

- `ZenohTransport` — cross-node pub/sub + queryable; handles bare topics (default)
- `LocalTransport` — in-process `queue.Queue`; used for tests
- `WebSocketTransport` — browser/UI connections; topic prefix `ws://`
- `IPCTransport` — inter-process on the same machine; topic prefix `ipc://`

### `Router` (router.py)

Owns an inbound queue and a dedicated router thread. Transport receiver threads
push `(topic, msg)` into the inbound queue. The router thread matches topics to
registered `TaskSpec`s, creates `Job`s, and calls `executor.enqueue()`.

Outbound (`publish`, `query`) is synchronous — called directly from worker
threads via `AciesContext`, no router thread involved.

Routing is prefix-based: `can_handle(topic)` on each transport determines which
one is used. Raises if no transport matches.

```python
class Router:
    def add(self, transport: Transport) -> None: ...
    def start(self, executor: Executor) -> None: ...  # starts transports + router thread
    def stop(self) -> None: ...

    # Inbound registration (called by AciesApp at startup)
    def subscribe(self, topic: str, spec: TaskSpec) -> None: ...
    def advertise(self, topic: str, spec: TaskSpec) -> None: ...

    # Outbound (called directly from worker threads via AciesContext)
    def publish(self, topic: str, msg: AciesMsg) -> None: ...
    def query(self, topic: str, msg: AciesMsg, timeout: float) -> AciesMsg | None: ...
```

### `Executor` (executor.py)

Owns an internal queue (ordering policy), a dispatcher thread, and a worker
thread pool. The router thread and timer thread call `enqueue()` to submit jobs.
After each handler returns, calls `job.reply_fn(result)` if set.

`start()` receives a `dispatch` callable provided by `AciesApp`. Keeping
dispatch as an injected callable means `Executor` has no knowledge of
`AciesContext`, avoiding a circular dependency.

```python
class Executor:
    def enqueue(self, job: Job) -> None: ...
    def start(self, dispatch: Callable[[Job], Any], n_workers: int = 4) -> None: ...
    def stop(self) -> None: ...
```

### `AciesApp` (app.py)

User-facing entry point. Owns `Router`, `Executor`, one shared timer thread, and
a `list[TaskSpec]` registry populated by the decorators. Each decorator
constructs the appropriate spec type directly — no `TaskKind` enum needed.

```python
class AciesApp:
    def __init__(self, name: str, router: Router | None = None): ...

    # Lifecycle hooks
    def on_startup(self, fn: Callable) -> Callable: ...
    def on_shutdown(self, fn: Callable) -> Callable: ...

    # Task decorators — each returns the appropriate spec type
    def subscribe(self, *topics: str) -> Callable: ...   # registers SubscriberSpec
    def schedule(self, interval: float) -> Callable: ... # registers ScheduleSpec
    def service(self, topic: str) -> Callable: ...       # registers ServiceSpec

    def run(self) -> None: ...
    def stop(self) -> None: ...   # sets stop event; can be called from anywhere
```

---

## 4. Main Loop

```python
def run(self):
    # Build one AciesContext per spec — reused across all jobs of that spec.
    # ctx.app is shared; ctx.task is isolated per spec.
    app_state = AppState()
    task_ctxs = {
        spec: AciesContext(
            publish_fn=self._router.publish,
            query_fn=self._router.query,
            app=app_state,
            task=TaskState(),
        )
        for spec in self._specs
    }

    # dispatch: O(1) ctx lookup; match on spec type for call convention
    def dispatch(job: Job) -> Any:
        ctx = task_ctxs[job.spec]
        match job.spec:
            case ScheduleSpec():
                return job.spec.fn(ctx)
            case SubscriberSpec() | ServiceSpec():
                return job.spec.fn(ctx, job.msg)

    self.executor.start(dispatch)
    self.router.start(self.executor)         # starts transport threads + router thread

    # startup hooks share a dedicated ctx backed by the same app_state
    startup_ctx = AciesContext(
        publish_fn=self._router.publish,
        query_fn=self._router.query,
        app=app_state,
        task=TaskState(),
    )
    for hook in self._startup_hooks:
        hook(startup_ctx)

    for spec in self._specs:
        match spec:
            case SubscriberSpec():
                for topic in spec.topics:
                    self._router.subscribe(topic, spec)
            case ServiceSpec():
                self._router.advertise(spec.topics[0], spec)
            case ScheduleSpec():
                pass  # handled by timer thread

    self._start_timer_thread()               # one thread, priority queue of (next_fire, spec)

    self._stop_event.wait()                  # main thread blocks here until stop() is called

    self.router.stop()
    self.executor.stop()

    for hook in self._shutdown_hooks:
        hook(startup_ctx)  # reuse the startup ctx for shutdown hooks
```

All three task kinds flow through the executor:

```
Transport thread → router inbound queue → router thread → executor.enqueue(Job(spec, msg))         [SUBSCRIBE]
Transport thread → router inbound queue → router thread → executor.enqueue(Job(spec, msg, reply))  [SERVICE]
Timer thread     →                                        executor.enqueue(Job(spec, msg=None))    [SCHEDULE]

Source threads (started in on_startup) call ctx.publish() → router.publish() → transport directly.
```

---

## 5. Minimal API (user-facing)

```python
import threading
from acies.corev2 import AciesApp, AciesContext, AciesMsg

app = AciesApp('my-node')

@app.on_startup
def setup(ctx: AciesContext):
    # Source thread: reads from hardware, publishes into the system
    def mic_thread():
        while True:
            raw = audio_driver.read()
            ctx.publish('sensors/audio', {'data': raw})
    threading.Thread(target=mic_thread, daemon=True).start()

# Timer-driven
@app.schedule(interval=1.0)
def poll_sensor(ctx: AciesContext):
    ctx.publish('sensors/temperature', {'value': read_hw_sensor()})

# Message-driven sink
@app.subscribe('sensors/temperature')
def log_temp(ctx: AciesContext, msg: AciesMsg):
    print(msg.payload)

# Message-driven transform
@app.subscribe('sensors/temperature')
def process(ctx: AciesContext, msg: AciesMsg):
    ctx.publish('processed/temperature', {'value': msg.payload['value'] * 1.8 + 32})

# Stateful handler: task-local accumulation across calls
@app.subscribe('sensors/temperature')
def accumulate(ctx: AciesContext, msg: AciesMsg):
    with ctx.task.lock:
        ctx.task.data.setdefault('readings', []).append(msg.payload['value'])

# Cross-handler: read a value written by another handler
@app.schedule(interval=5.0)
def report(ctx: AciesContext):
    with ctx.app.lock:
        last = ctx.app.data.get('last_temp')
    if last is not None:
        ctx.publish('reports/temp', {'last': last})

# RPC: return value is sent as reply
@app.service('rpc/status')
def status(ctx: AciesContext, msg: AciesMsg) -> dict:
    return {'node': app.name, 'status': 'ok'}

@app.on_shutdown
def teardown(ctx: AciesContext):
    pass

app.run()
```

---

## 6. Testing Plan

### Unit tests (no Zenoh required)

| File               | What it tests                                                                               |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `test_task.py`     | `TaskSpec` construction, `Job` creation, `reply_fn` for SERVICE jobs                        |
| `test_executor.py` | Jobs run in worker threads; `reply_fn` called after handler returns; FIFO ordering          |
| `test_router.py`   | Prefix routing; exception on no matching transport; pub/sub roundtrip with `LocalTransport` |

### Integration tests

| File          | What it tests                                                                                                                                        |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_app.py` | Full `AciesApp` lifecycle with `LocalTransport`: startup hook, `@schedule`, `@subscribe`, `@service` reply, `ctx.task`/`ctx.app` state, `stop()`, shutdown hook |

**Transport strategy for tests**: `AciesApp` accepts an optional `router=`
argument. Tests construct a `Router` with `LocalTransport` and pass it in — no
Zenoh process needed, fully deterministic.

### Running tests

```bash
uv run pytest packages/core/tests/corev2/         # all corev2 tests
uv run pytest packages/core/tests/corev2/ -x -s   # stop on first failure, show output
```

---

## 7. Implementation Phases

### Phase 1: Skeleton ✓ (complete)

**Goal**: All files created, core data structures fully implemented, everything
else stubbed. The thread model and scheduling semantics are reflected in the
code structure even before any thread runs.

1. Create `packages/core/src/acies/corev2/` with all 8 files
2. `msg.py` — placeholder stub
3. `task.py` — fully implement:
   - `SubscriberSpec`, `ScheduleSpec`, `ServiceSpec` frozen dataclasses
   - `TaskSpec = SubscriberSpec | ScheduleSpec | ServiceSpec` type alias
   - `Job` dataclass: `spec`, `msg`, `created_at`, `reply_fn` (only set for `ServiceSpec` jobs)
4. `transport.py` — stub `Transport` Protocol with full interface:
   `start(on_message)`, `stop()`, `can_handle(topic)`, `publish()`, `subscribe()`,
   `query()`, `advertise()`; stub `LocalTransport` as a concrete implementation
5. `router.py` — stub `Router`: owns an inbound queue and a router thread;
   exposes `add(transport)`, `start(executor)`, `stop()`, `subscribe(topic, spec)`,
   `advertise(topic, spec)`, `publish(topic, msg)`, `query(topic, msg, timeout)`
6. `executor.py` — stub `Executor`: owns an internal queue, a dispatcher thread,
   and a worker pool; exposes `enqueue(job)`, `start(dispatch, n_workers)`, `stop()`;
   FIFO is the initial policy (extension point for future scheduling policies)
7. `context.py` — stub `AppState`, `TaskState` (lock + data dict); stub
   `AciesContext`: `publish()`, `query()`, `app`, `task`
8. `app.py` — stub `AciesApp`: decorators (`subscribe`, `schedule`, `service`,
   `on_startup`, `on_shutdown`) register `TaskSpec`s; `run()` and `stop()` stubbed
9. `__init__.py` — export public API: `AciesApp`, `AciesContext`, `AciesMsg`,
   `TaskSpec`, `SubscriberSpec`, `ScheduleSpec`, `ServiceSpec`

**Done when**: `from acies.corev2 import AciesApp` works; decorators register
the correct spec type into `app._specs`; all classes instantiate without error.

---

### Phase 2: Tests

**Goal**: Write tests before implementation to define expected behaviour and
catch API design issues early.

1. Write `test_task.py` — pure data structure tests, no threads
2. Write `test_executor.py` — threading, dispatch, reply_fn, FIFO ordering
3. Write `test_router.py` — routing, LocalTransport roundtrip
4. Write `test_app.py` — full lifecycle integration with LocalTransport

**Done when**: All tests exist and fail cleanly (not error) against the stubs.

---

### Phase 3: Runtime

**Goal**: Messages flow end-to-end using `LocalTransport`; all Phase 2 tests pass.

1. Implement `LocalTransport` (in-process queues, receiver thread)
2. Implement `Router` — inbound queue, router thread, prefix routing, exception on no match
3. Implement `Executor` — internal FIFO queue, dispatcher thread, worker pool, `reply_fn` handling
4. Implement `AppState`, `TaskState`, `AciesContext` — `publish()`, `query()`, `app`, `task`
5. Implement shared timer thread in `AciesApp` using a priority queue
6. Wire `AciesApp.run()` and `stop()` as sketched in §4 (per-spec ctx construction, two-branch dispatch)

**Done when**: `uv run pytest packages/core/tests/corev2/` passes.

---

### Phase 4: ZenohTransport

**Goal**: Works with a real Zenoh session.

1. Implement `ZenohTransport` (zenoh 1.x pub/sub + queryable)
2. `advertise` passes a `reply_fn` closure into the `Job` via the router
3. Smoke test: two `AciesApp` instances, one publishes, one subscribes

**Done when**: Cross-process pub/sub and service RPC work over Zenoh.

---

### Phase 5: Refactor `acies.controller`

**Goal**: `Controller` reimplemented as an `AciesApp` with decorators.

| Current (`Service` subclass)            | v2 equivalent                      |
| --------------------------------------- | ---------------------------------- |
| `__init__` + manual `subscribe()` calls | `@app.subscribe(topic)` decorators |
| Periodic `schedule()` calls             | `@app.schedule(interval=...)`      |
| External sensor callbacks               | source thread started in `@app.on_startup` |
| `make_msg()` + `publish()`              | `ctx.publish(topic, payload=...)`  |
| Zenoh queryable setup                   | `@app.service(topic)`              |
| Startup logic in `__init__`             | `@app.on_startup`                  |

Steps:

1. Add `acies-corev2` dependency to `packages/controller/pyproject.toml`
2. Rewrite `base.py` using `AciesApp`
3. Keep `state.py`, `analysis.py`, `ns.py`, `buffer.py` unchanged
4. Update `__init__.py` exports
5. Smoke test: run controller against a live Zenoh router

---

### Phase 6: Refactor `acies.vehicle_classifier`

**Goal**: `Classifier` base class reimplemented using `AciesApp`.

Steps:

1. Add `acies-corev2` dependency
2. Rewrite `base.py`: `load_model()` and `infer()` remain abstract; sensor
   subscription becomes `@app.subscribe(mic_topic, geo_topic)`; inference trigger
   becomes `@app.schedule(interval=...)` or a source thread in `@app.on_startup`
3. Update concrete classifiers (`vfm.py`, `ds.py`, `simple.py`) to use new base
4. Keep model integration files (`deepsense.py`, `foundationsense.py`) unchanged
5. Smoke test with recorded sensor data

---

### Phase 7: Cleanup

1. Remove `acies.core` imports from controller and vehicle-classifier
2. Deprecation notice in `acies/core/__init__.py`
3. Update docs
4. Final full test run across all packages
