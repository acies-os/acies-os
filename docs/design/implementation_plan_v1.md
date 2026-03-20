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

### `Transport` (transport.py)

Protocol defining the interface for messaging backends. Backends implement the
protocol structurally — no inheritance required. Each backend owns its receiver
thread(s) and pushes received messages into the router's inbound queue via a
callback.

```python
class Transport(Protocol):
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

Concrete backends (satisfy `Transport` structurally):

- `ZenohTransport` — cross-node pub/sub + queryable; primary production backend.
  Same-host IPC is handled by configuring the session with a Unix domain socket
  endpoint or enabling Zenoh SHM — no separate transport needed.
- `WebSocketTransport` — browser/UI connections; runs its own WebSocket
  server thread; bridges connected clients to the router's `on_message` callback.
  Topic prefix `ws://` routes outbound messages to this transport.
- `LocalTransport` — in-process queue; used for tests and single-process apps.

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

| File          | What it tests                                                                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
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

### Phase 1: Skeleton (update to latest design)

**Goal**: All files reflect the current design — sum type specs, `AppState`/
`TaskState`, no `TaskKind` enum, no `PRODUCE`. Everything except `task.py` and
`msg.py` remains a stub; thread wiring is deferred to Phase 3.

Files and what changes:

| File                                       | Change                                                                                                                       |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `task.py`                                  | Replace `TaskKind` + single `TaskSpec` with `SubscriberSpec`, `ScheduleSpec`, `ServiceSpec` sum types; update `Job` comments |
| `msg.py`                                   | Add `topic: str` field (set by router at delivery time — handlers need it to distinguish topics when subscribed to multiple) |
| `context.py`                               | Add `AppState`, `TaskState` (lock + data dict); update `AciesContext` to carry `app` and `task`; remove `msg()` helper       |
| `app.py`                                   | Remove `produce` decorator and `_specs_of`; decorators create the correct spec type                                          |
| `__init__.py`                              | Export `SubscriberSpec`, `ScheduleSpec`, `ServiceSpec`, `AppState`, `TaskState`; remove `TaskKind`                           |
| `executor.py`, `router.py`, `transport.py` | No change — stubs remain                                                                                                     |

**Done when**: `from acies.corev2 import AciesApp, SubscriberSpec` works; all
classes instantiate; decorators produce the correct spec type.

---

### Phase 2: API Sketches and Reference Applications

**Goal**: Draft the new API by writing skeleton implementations of
`packages/sensors/` and `packages/vehicle-classifier/` using `AciesApp`.
These sketches surface design issues before tests are written, while the cost
of changing the API is still low.

Deliverables:

- `packages/sensors/mic.py` — mic sensor as an `AciesApp`: source thread in
  `on_startup`, `@service` for runtime config
- `packages/sensors/geo.py` — same pattern for geophone
- `packages/vehicle-classifier/.../classifier_v2.py` — `Classifier` base
  reimplemented: `@subscribe` for sensor data, `@schedule` for inference,
  `@service` for status/config; `load_model` and `infer` remain abstract

Known design issues to resolve during this phase:

1. **`msg.topic`** — handlers subscribed to multiple topics (e.g. `geo` and
   `mic`) need to know which topic a message came from. Already added to
   `AciesMsg` in Phase 1; confirm usage here.
2. **Cross-handler shared state** — the sensor buffer is written by
   `handle_sensor` and read by `run_inference`. These are different specs with
   different `ctx.task` stores. Shared data must go in `ctx.app.data`. Establish
   the pattern: `ctx.task` for handler-private state, `ctx.app` for
   inter-handler coordination.
3. **Wildcard topics** — `sensors/+/geo` style patterns needed by the
   classifier. Confirm the `Transport` Protocol and `Router` must support
   zenoh-style wildcards (`+`, `**`). `LocalTransport` needs basic wildcard
   matching for tests.
4. **App-per-node vs app-per-process** — the classifier mixes sensor
   subscriptions and inference scheduling in one app. Confirm this is the
   intended model (one `AciesApp` per logical node).

**Done when**: Sketches compile and read clearly as the intended API; all
design issues from the list above are resolved and documented.

---

### Phase 3: Tests

**Goal**: Codify the API from Phase 2 as executable tests. Tests are written
against the stubs from Phase 1 — they will fail, but must not error.

```
packages/core/tests/corev2/
    ├── __init__.py
    ├── test_task.py       # spec construction, immutability, Job fields
    ├── test_executor.py   # dispatch in worker thread, reply_fn, FIFO order
    ├── test_router.py     # prefix routing, no-transport error, pub/sub roundtrip
    └── test_app.py        # full lifecycle: startup, subscribe, schedule, service, stop
```

Unit test coverage:

| File               | Key cases                                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_task.py`     | Each spec type construction and frozen; `Job` defaults; `reply_fn` only on service jobs                                                                         |
| `test_executor.py` | Handler runs in worker thread (not main); `reply_fn(result)` called after dispatch returns; FIFO with `n_workers=1`; `stop()` after drain                       |
| `test_router.py`   | `RuntimeError` with no transport; first-match routing; `subscribe` + `publish` delivers job; multiple subscribers on same topic; service job carries `reply_fn` |

Integration test (`test_app.py` with `LocalTransport`):

| Case                   | What it checks                                                |
| ---------------------- | ------------------------------------------------------------- |
| startup hook           | runs before `run()` blocks; `ctx` is live                     |
| `@subscribe`           | message published inside app reaches handler                  |
| `@schedule`            | handler fires ~N times in N×interval seconds                  |
| `@service`             | `ctx.query()` returns handler's return value                  |
| `ctx.app` / `ctx.task` | state persists across calls; app state shared across handlers |
| `stop()`               | unblocks `run()`; shutdown hook runs                          |

**Done when**: All test files exist; `uv run pytest packages/core/tests/corev2/ --collect-only`
succeeds; tests fail (not error) against the stubs.

---

### Phase 4: Runtime

**Goal**: All Phase 3 tests pass using `LocalTransport`.

1. `msg.py` — finalize `AciesMsg` (topic field, repr)
2. `context.py` — implement `AppState`, `TaskState`; implement `AciesContext.publish()` and `query()`
3. `transport.py` — implement `LocalTransport` (in-process queue, receiver thread, basic wildcard matching)
4. `router.py` — implement inbound queue, router thread, topic→spec matching, `reply_fn` injection for service jobs
5. `executor.py` — implement FIFO queue, dispatcher thread, worker pool, `reply_fn` call after dispatch
6. `app.py` — implement `run()`: per-spec ctx construction, two-branch dispatch, timer thread, wiring loop; implement `stop()`

**Done when**: `uv run pytest packages/core/tests/corev2/` is fully green.

---

### Phase 5: Production Transports

**Goal**: Works with real backends.

1. `ZenohTransport` — zenoh 1.x pub/sub + queryable; `reply_fn` closure via `advertise`
2. `WebSocketTransport` — WebSocket server thread; `ws://` topic prefix; bridges browser clients to router
3. Smoke test: two `AciesApp` instances over Zenoh (pub/sub + service RPC)
4. Smoke test: browser client receives a published message over WebSocket

**Done when**: Cross-process pub/sub, service RPC, and WebSocket delivery all work.

---

### Phase 6: Refactor Application Packages

**Goal**: `acies.sensors`, `acies.controller`, and `acies.vehicle_classifier`
all use the new `AciesApp` API. Phase 2 sketches are the starting point.

| Package                    | Key changes                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| `acies.sensors`            | `mic.py`, `geo.py` become `AciesApp` instances; source threads in `on_startup`                            |
| `acies.controller`         | `base.py` rewritten; `state.py`, `analysis.py`, `ns.py`, `buffer.py` unchanged                            |
| `acies.vehicle_classifier` | `base.py` rewritten from Phase 2 sketch; `load_model`/`infer` stay abstract; concrete classifiers updated |

Steps for each package:

1. Add `acies-corev2` dependency to `pyproject.toml`
2. Rewrite using Phase 2 sketch as the blueprint
3. Smoke test against a live Zenoh router

**Done when**: All three packages run end-to-end against Zenoh.

---

### Phase 7: Cleanup

1. Remove `acies.core` imports from all refactored packages
2. Add deprecation notice to `acies/core/__init__.py`
3. Final full test run: `uv run pytest`
4. Update user-facing docs
