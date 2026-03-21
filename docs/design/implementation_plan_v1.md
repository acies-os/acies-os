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
    └── msg.py              # control messages + built-in data types (AciesTensor, etc.)
```

Tests:

```
packages/core/tests/corev2/
    ├── __init__.py
    ├── test_executor.py
    ├── test_transport.py
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
    msg_type: type | None = None  # extracted from fn annotation at decoration time

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
    msg_type: type | None = None  # extracted from fn annotation at decoration time

TaskSpec = SubscriberSpec | ScheduleSpec | ServiceSpec
```

`msg_type` is extracted by inspecting the `msg` parameter annotation of the
handler at decoration time. The router uses it as the decode target:
`msgspec.msgpack.decode(raw, type=spec.msg_type)`. Falls back to
`msgspec.msgpack.decode(raw)` if `None`.

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
    raw: bytes | None                         # raw msgpack bytes; None for ScheduleSpec jobs
    created_at: float                         # time.monotonic()
    reply_fn: ReplyFn | None                  # only set for ServiceSpec jobs; dispatch calls reply_fn(encoded_bytes)
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

    def publish(self, topic: str, msg: msgspec.Struct) -> None: ...
    def query(self, topic: str, msg: msgspec.Struct, timeout: float = 1.0) -> msgspec.Struct | None: ...
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
    def start(self, on_message: Callable[[str, bytes], None]) -> None: ...
    def stop(self) -> None: ...
    def can_handle(self, topic: str) -> bool: ...
    def publish(self, topic: str, raw: bytes) -> None: ...
    def subscribe(self, topic: str) -> None: ...
    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None: ...
    def advertise(self, topic: str, reply_fn_factory: Callable) -> None: ...
```

The transport layer deals exclusively in raw bytes — it has no knowledge of
message types. Encoding (`msgspec.msgpack.encode`) and decoding
(`msgspec.msgpack.decode`) happen in the router:

- **Inbound**: transport calls `on_message(topic, raw_bytes)`; router decodes
  using `spec.msg_type`.
- **Outbound** (`publish`/`query`): router encodes the `msgspec.Struct` to
  bytes before calling `transport.publish` / `transport.query`.

`start()` receives an `on_message` callback — the transport calls it from its
receiver thread whenever a message arrives, passing `(topic, raw_bytes)` to the
router.

**Wildcard matching rules** (zenoh-style, required for `LocalTransport` in tests):

- `+` matches exactly one path segment (e.g. `sensors/+/geo` matches
  `sensors/unit1/geo` but not `sensors/geo`)
- `**` matches zero or more path segments (e.g. `sensors/**` matches
  `sensors/mic`, `sensors/unit1/geo`, etc.)
- Exact match always works

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
    def publish(self, topic: str, msg: msgspec.Struct) -> None: ...
    def query(self, topic: str, msg: msgspec.Struct, timeout: float) -> msgspec.Struct | None: ...
```

Inbound flow in `_route_loop`:
1. Receive `(topic, raw_bytes)` from the inbound queue.
2. If topic starts with `acies/ctrl/`: handle control message internally (never
   creates a Job).
3. Otherwise: look up matching `SubscriberSpec`s / `ServiceSpec`; decode
   `msgspec.msgpack.decode(raw, type=spec.msg_type)` (or untyped fallback);
   create `Job(spec, decoded_msg)` and call `executor.enqueue(job)`.

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
Transport thread → router inbound queue → router thread → decode → executor.enqueue(Job(spec, msg))         [SUBSCRIBE]
Transport thread → router inbound queue → router thread → decode → executor.enqueue(Job(spec, msg, reply))  [SERVICE]
Timer thread     →                                                  executor.enqueue(Job(spec, msg=None))    [SCHEDULE]

Source threads (started in on_startup) call ctx.publish(msg) → router encodes → transport.publish(raw).
```

---

## 5. Minimal API (user-facing)

```python
import threading
import msgspec
from acies.corev2 import AciesApp, AciesContext

app = AciesApp('my-node')

# User-defined message types (msgspec.Struct, frozen=True)
class TempReading(msgspec.Struct, frozen=True):
    source: str
    timestamp: int
    value: float

class StatusReply(msgspec.Struct, frozen=True):
    source: str
    ok: bool

@app.on_startup
def setup(ctx: AciesContext):
    # Source thread: reads from hardware, publishes into the system
    def mic_thread():
        while True:
            raw = audio_driver.read()
            ctx.publish('sensors/audio', AciesTensor(source='my-node', timestamp=now_ns(), payload=raw))
    threading.Thread(target=mic_thread, daemon=True).start()

# Timer-driven
@app.schedule(interval=1.0)
def poll_sensor(ctx: AciesContext):
    ctx.publish('sensors/temperature', TempReading(source='my-node', timestamp=now_ns(), value=read_hw_sensor()))

# Message-driven sink — msg type annotation drives decoding
@app.subscribe('sensors/temperature')
def log_temp(ctx: AciesContext, msg: TempReading):
    print(msg.value)

# Message-driven transform
@app.subscribe('sensors/temperature')
def process(ctx: AciesContext, msg: TempReading):
    ctx.publish('processed/temperature', TempReading(source='my-node', timestamp=msg.timestamp, value=msg.value * 1.8 + 32))

# Stateful handler: task-local accumulation across calls
@app.subscribe('sensors/temperature')
def accumulate(ctx: AciesContext, msg: TempReading):
    with ctx.task.lock:
        ctx.task.data.setdefault('readings', []).append(msg.value)

# Cross-handler: read a value written by another handler
@app.schedule(interval=5.0)
def report(ctx: AciesContext):
    with ctx.app.lock:
        last = ctx.app.data.get('last_temp')
    if last is not None:
        ctx.publish('reports/temp', TempReading(source='my-node', timestamp=now_ns(), value=last))

# RPC: return value is sent as reply
@app.service('rpc/status')
def status(ctx: AciesContext, msg: msgspec.Struct) -> StatusReply:
    return StatusReply(source=app.name, ok=True)

@app.on_shutdown
def teardown(ctx: AciesContext):
    pass

app.run()
```

---

## 6. Testing Plan

### Unit tests (no Zenoh required)

**`test_executor.py`** (T2):

| Case                                                    | How to verify                                                                                         |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Handler runs in a worker thread, not the calling thread | Capture `threading.current_thread()` inside handler; assert it differs from `threading.main_thread()` |
| `reply_fn(result)` called with handler's return value   | Pass a `reply_fn` that appends to a list; assert list contains the return value after job completes   |
| FIFO ordering with `n_workers=1`                        | Enqueue 5 jobs recording arrival order; assert execution order matches                                |
| `stop()` after all jobs complete                        | Enqueue jobs; call `stop()`; assert no jobs are lost                                                  |

**`test_transport.py`** (T3):

| Case                                                           | How to verify                                                       |
| -------------------------------------------------------------- | ------------------------------------------------------------------- |
| `publish` triggers `on_message` with correct topic and raw bytes | Use `threading.Event` to synchronize                               |
| Unsubscribed topic is not delivered                            | Publish to two topics; subscribe to one; assert only one delivery   |
| Wildcard `sensors/+/geo` matches `sensors/unit1/geo`           | Subscribe with pattern; publish to matching topic                   |
| Wildcard `sensors/+/geo` does not match `sensors/geo`          | Publish; assert no delivery within short timeout                    |
| `query` returns the reply                                      | Advertise handler that returns a value; call `query`; assert result |
| `query` returns `None` on timeout                              | No advertiser registered; `query` with short timeout                |

**`test_router.py`** (T4):

| Case                                                  | How to verify                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `RuntimeError` with no transports                     | Call `router.publish(...)` before adding any transport                             |
| First-match routing                                   | Add two LocalTransports with different `can_handle` logic; verify correct one used |
| `subscribe` + `publish` delivers job to executor      | Publish a message; assert handler was called                                       |
| Multiple subscribers on same topic each receive a job | Two specs subscribed to same topic; publish once; both execute                     |
| Service job carries `reply_fn`                        | `ctx.query()` returns handler's return value end-to-end                            |

### Integration tests

**`test_app.py`** (T5) — all cases use `LocalTransport` via injected `Router`:

| Case                   | What to check                                                    |
| ---------------------- | ---------------------------------------------------------------- |
| `on_startup` hook      | Runs before `run()` blocks; `ctx.publish()` works from inside it |
| `@subscribe`           | Message published inside the app reaches the handler             |
| `@schedule`            | Handler fires approximately N times in N × interval seconds      |
| `@service`             | `ctx.query(topic)` returns the handler's return value            |
| `ctx.task` persistence | Value written in call N is readable in call N+1 for same spec    |
| `ctx.app` sharing      | Value written by one handler is readable by a different handler  |
| `stop()`               | Unblocks `run()`; shutdown hook runs after                       |

**Transport strategy for tests**: `AciesApp` accepts an optional `router=`
argument. Tests construct a `Router` with `LocalTransport` and pass it in — no
Zenoh process needed, fully deterministic. Run the app in a background thread
in each test; call `app.stop()` to terminate; join with a timeout.

### Running tests

```bash
uv run pytest packages/core/tests/corev2/         # all corev2 tests
uv run pytest packages/core/tests/corev2/ -x -s   # stop on first failure, show output
```

---

## 7. Implementation Tasks

Tasks are ordered by a dependency graph, not sequentially. Tasks without shared
dependencies may be handed off in parallel.

### Dependency Graph

```
T1 (types/skeleton)
├── T2 (Executor)       ──────────────────────────────────┐
└── T3 (LocalTransport) ─────────────────────────────── T4 (Router) ── T5 (AciesApp.run)
                                                                             ├── T6 (sensors)
                                                                             ├── T7 (ZenohTransport)
                                                                             │    ├── T8 (sensors hw)
                                                                             │    ├── T9 (classifier)
                                                                             │    └── T10 (controller)
                                                                             └── T11 (cleanup, after T8–T10)
```

T2 and T3 may be handed off in **parallel** after T1 completes.

---

### T1 — Fix corev2 skeleton to match design_v2

**Goal**: All corev2 source files reflect the current design. No runtime
behavior is implemented — stubs remain stubs. Only types, dataclasses, and the
public API surface change.

| File          | What changes                                                                                       |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `task.py`     | Replace `TaskKind` enum and single `TaskSpec` with three frozen dataclasses; update `Job`          |
| `msg.py`      | Add `topic: str = ""` field (set by router at delivery time)                                       |
| `context.py`  | Add `AppState` and `TaskState`; update `AciesContext.__init__`; remove `msg()` helper              |
| `app.py`      | Remove `produce` decorator and `_specs_of`; add `router: Router \| None = None` to `__init__`      |
| `__init__.py` | Export `SubscriberSpec`, `ScheduleSpec`, `ServiceSpec`, `AppState`, `TaskState`; remove `TaskKind` |

**Verification**:

```bash
uv run python -c "
from acies.corev2 import (
    AciesApp, AciesContext, Job,
    SubscriberSpec, ScheduleSpec, ServiceSpec,
    AppState, TaskState,
)
app = AciesApp('test')

@app.subscribe('a', 'b')
def h1(ctx, msg): pass

@app.schedule(interval=1.0)
def h2(ctx): pass

@app.service('rpc/x')
def h3(ctx, msg): pass

specs = app._specs
assert isinstance(specs[0], SubscriberSpec), specs[0]
assert isinstance(specs[1], ScheduleSpec), specs[1]
assert isinstance(specs[2], ServiceSpec), specs[2]
assert specs[0].topics == ('a', 'b')
assert specs[1].interval == 1.0
print('ok')
"
```

---

### T2 — Implement Executor + tests

**Dependencies**: T1

**Goal**: `Executor` fully implemented and tested. No transport or router
required — tests inject jobs directly via `enqueue()`.

**Files**: `executor.py` (implement); `tests/corev2/__init__.py` (empty, create);
`tests/corev2/test_executor.py` (create)

Behavior:

- `start(dispatch, n_workers=4)`: start `ThreadPoolExecutor`; start dispatcher thread running `_dispatch_loop`
- `enqueue(job)`: put `job` into `self._queue`
- `_dispatch_loop`: drain queue; submit each job to pool; exit when stop event set and queue empty
- `_run_job(job)`: call `self._dispatch(job)`; dispatch is responsible for calling `job.reply_fn` if set
- `stop()`: set stop event; put sentinel to unblock dispatcher; join thread; shutdown pool with `wait=True`

**Verification**:

```bash
uv run pytest packages/core/tests/corev2/test_executor.py -x -s
```

---

### T3 — Implement LocalTransport + tests

**Dependencies**: T1

**Goal**: `LocalTransport` fully implemented and tested in isolation.

**Files**: `transport.py` (implement `LocalTransport`; leave `Transport` Protocol
unchanged); `tests/corev2/test_transport.py` (create)

Behavior:

- `start(on_message)`: store callback; start receiver thread draining internal queue; call `on_message(topic, raw_bytes)` per item
- `stop()`: put sentinel; join thread
- `subscribe(topic)`: add pattern to active subscriptions set
- `can_handle(topic)`: always `True`
- `publish(topic, raw)`: if topic matches any subscription pattern, put `(topic, raw)` into queue
- `advertise(topic, reply_fn_factory)`: register queryable
- `query(topic, raw, timeout)`: put query into queue; block on `threading.Event`; return reply bytes or `None`

Wildcard matching follows the rules in §3 Transport above.

**Verification**:

```bash
uv run pytest packages/core/tests/corev2/test_transport.py -x -s
```

---

### T4 — Implement Router + tests

**Dependencies**: T2, T3

**Goal**: `Router` fully implemented and tested using `LocalTransport` and a
real `Executor`.

**Files**: `router.py` (implement); `tests/corev2/test_router.py` (create)

Behavior:

- `add(transport)`: append to `self._transports`
- `start(executor)`: start each transport with `_on_message` callback; start router thread running `_route_loop(executor)`
- `_on_message(topic, raw)`: put `(topic, raw_bytes)` into `self._inbound`
- `_route_loop(executor)`: for each `(topic, raw_bytes)`:
  - If `topic` starts with `acies/ctrl/`: handle control message internally (never creates a Job)
  - Otherwise: decode `msgspec.msgpack.decode(raw, type=spec.msg_type)` for each matching spec; create `Job(spec, decoded_msg)` or `Job(spec, decoded_msg, reply_fn=...)` for SERVICE; call `executor.enqueue(job)`
- `subscribe(topic, spec)`: add to `_subscriptions`; call `transport.subscribe(topic)` on matching transports
- `advertise(topic, spec)`: add to `_services`; call `transport.advertise(topic, reply_fn_factory)` on matching transport
- `stop()`: set stop event; put sentinel; join thread; stop each transport
- `publish(topic, msg)`: encode `msgspec.msgpack.encode(msg)`; delegate bytes to `_transport_for(topic)`; raise `RuntimeError` if no match
- `query(topic, msg, timeout)`: encode; delegate to transport; decode reply bytes if not `None`

**Verification**:

```bash
uv run pytest packages/core/tests/corev2/test_router.py -x -s
```

---

### T5 — Implement AciesApp.run() + integration test

**Dependencies**: T2, T3, T4

**Goal**: `AciesApp.run()` fully implemented; full lifecycle integration test
passes with `LocalTransport`.

**Files**: `app.py` (implement `run()`, `_timer_loop()`, `stop()`);
`tests/corev2/test_app.py` (create)

`_timer_loop()`: use `heapq` priority queue of `(next_fire_time, spec)`. Sleep
until next fire time; call `self._executor.enqueue(Job(spec, msg=None))`; reschedule.
Exit when stop event is set.

`__init__` change: if `router` argument provided, use it; otherwise construct
`Router()` with a default `LocalTransport` added.

**Verification**:

```bash
uv run pytest packages/core/tests/corev2/ -x -s
```

All four test files must pass.

---

### T6 — Implement sensors package (mic + geo)

**Dependencies**: T5

**Goal**: `mic.py` and `geo.py` are real `AciesApp` instances within the
proper package structure. These become the production implementations in T8.

**Package structure**:

```
packages/sensors/
├── pyproject.toml
└── src/acies/sensors/
    ├── __init__.py
    ├── mic.py
    └── geo.py
```

Move existing loose `packages/sensors/mic.py` and `geo.py` into
`src/acies/sensors/`. Update `pyproject.toml`: rename to `acies-sensors`, set
source layout, replace `acies-core` dependency with `acies-corev2` (workspace).

Each module follows the pattern: `app = AciesApp('acies-mic')` with source
thread in `on_startup` and `@service` for runtime config.

**Verification**:

```bash
uv run python -c "from acies.sensors import mic, geo; print('ok')"
```

---

### T7 — Implement ZenohTransport

**Dependencies**: T5

**Goal**: `ZenohTransport` works as a drop-in replacement for `LocalTransport`.

**Files**: `zenoh_transport.py` (create); `tests/corev2/test_zenoh_smoke.py` (create)

Behavior mirrors `LocalTransport` using a zenoh 1.x session. `can_handle(topic)`
returns `True` for topics not starting with `ws://`. Mark smoke tests with
`pytest.mark.skipif` if zenoh router is not reachable.

**Verification**:

```bash
uv run pytest packages/core/tests/corev2/test_zenoh_smoke.py -x -s
```

---

### T8 — Wire sensors to hardware drivers

**Dependencies**: T6, T7

**Goal**: `mic.py` and `geo.py` use real hardware drivers; package runs on
device without import errors.

Replace stub thread bodies with real driver calls. Update `pyproject.toml`
dependencies as needed.

**Verification**:

```bash
uv run python -m acies.sensors.mic   # starts without error; Ctrl-C to stop
```

---

### T9 — Refactor acies-vehicle-classifier

**Dependencies**: T5, T7

**Goal**: `Classifier` base class rewritten using `@subscribe`, `@schedule`,
`@service`. Concrete classifiers updated. `acies.core` imports removed.

Key patterns: `@app.subscribe('sensors/+/geo', 'sensors/+/mic')` buffers into
`ctx.app.data`; `@app.schedule(interval=...)` reads buffer and runs `self.infer()`;
`load_model` and `infer` remain abstract.

**Verification**:

```bash
uv run pytest packages/vehicle-classifier/tests/ -x
```

---

### T10 — Refactor acies-controller

**Dependencies**: T5, T7

**Goal**: `base.py` rewritten using `AciesApp`. `state.py`, `analysis.py`,
`ns.py`, `buffer.py` unchanged. `acies.core` imports removed.

**Verification**:

```bash
uv run pytest packages/controller/tests/ -x
```

---

### T11 — Cleanup

**Dependencies**: T8, T9, T10

**Goal**: Remove old middleware; full test suite green.

1. Add deprecation notice to `packages/core/src/acies/core/__init__.py`:
   ```python
   import warnings
   warnings.warn(
       "acies.core is deprecated; use acies.corev2",
       DeprecationWarning,
       stacklevel=2,
   )
   ```
2. Confirm no remaining `from acies.core import` or `import acies.core` in
   `packages/sensors/`, `packages/vehicle-classifier/`, `packages/controller/`
3. Run full suite

**Verification**:

```bash
uv run pytest
```
