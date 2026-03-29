"""AciesApp — user-facing entry point.

Owns the Router, Executor, timer thread, and TaskSpec registry.
Decorators register TaskSpecs; run() wires everything together and blocks.

Thread model:
  Main thread       — runs run(); blocks on _stop_event after wiring
  Transport threads — owned by each Transport; push to router inbound queue
  Router thread     — drains inbound queue; creates Jobs; calls executor.enqueue()
  Timer thread      — one shared thread; fires SCHEDULE jobs into executor
  Dispatcher thread — owned by Executor; drains internal queue; submits to pool
  Worker threads    — execute handlers; call ctx.publish() synchronously
  Managed threads   — @app.thread fns; started after startup hooks, joined before shutdown hooks
"""

from __future__ import annotations

import heapq
import inspect
import logging
import socket
import threading
import time
import uuid
from typing import Any, Callable, get_type_hints

import msgspec

from ._cli import create_acies_cli
from ._control import make_heartbeat_spec, make_io_spec, make_kv_spec, make_route_spec, make_schema_spec
from .context import AciesContext, AppState, TaskState, deep_merge
from .executor import Executor
from .namespace import CtlTopic, Namespace, Topic, TopicArg
from .router import Router
from .signal_handlers import temporary_signal_handlers
from .task import Job, ScheduleSpec, ServiceSpec, SubscriberSpec, TaskSpec, ThreadSpec
from .transport import ZenohTransport

logger = logging.getLogger(__name__)


def _get_msg_encoding_metadata(t: type) -> dict[str, str | bool] | None:
    """Return encoding metadata for a type if it is a msgspec.Struct subclass."""
    assert isinstance(t, type), f'expected a type, got {t!r}'
    if issubclass(t, msgspec.Struct):
        return {'format': 'msgpack', 'array_like': t.__struct_config__.array_like}
    return None


class AciesApp:
    def __init__(
        self,
        name: str | None = None,
        host: str | None = None,
        router: Router | None = None,
    ) -> None:
        resolved_name = name or uuid.uuid4().hex[:6]
        resolved_host = host or socket.gethostname()
        if router is not None:
            self._router: Router = router
        else:
            self._router = Router()
            self._router.add_transport(ZenohTransport())
        self._executor: Executor = Executor()
        self._tasks: list[TaskSpec] = []
        self._startup_hooks: list[Callable[..., None]] = []
        self._shutdown_hooks: list[Callable[..., None]] = []
        self._stop_event: threading.Event = threading.Event()
        self._timer_thread: threading.Thread | None = None
        self._task_ctxs: dict[TaskSpec, AciesContext] = {}
        self._app_state: AppState = AppState()
        self._app_state.config['sys'] = {
            'host': resolved_host,
            'name': resolved_name,
            'state': 'initializing',
        }
        self._ns: Namespace  # initialized in run() from config['sys']

        # ---------------------------- system tasks ----------------------------
        # periodic heartbeat
        self._tasks.append(make_heartbeat_spec())
        # key-value operations (set/get/del) on app_state.config
        self._tasks.append(make_kv_spec())
        # dynamic routing: task inputs/outputs topic updates
        self._tasks.append(make_route_spec(self._router))
        # task graph introspection: mapping of tasks to their input and output topics
        self._tasks.append(make_io_spec(self._router))
        # service schema introspection: request/response types for registered services
        self._tasks.append(make_schema_spec())

    @property
    def name(self) -> str:
        return self._app_state.config['sys']['name']  # type: ignore[no-any-return]

    @property
    def host(self) -> str:
        return self._app_state.config['sys']['host']  # type: ignore[no-any-return]

    @property
    def state(self) -> AppState:
        return self._app_state

    # --------------------------------- CLI -----------------------------------

    def cli(self, **kwargs: Any) -> Callable[[Callable[..., None]], Callable[..., None]]:
        """Decorator that turns a function into a Click command with middleware options.

        Middleware options (--acies-host, --acies-name) are injected and consumed
        before the user's function runs; they never appear in the user's kwargs.
        Parsed values are stored in app_state.config['sys'] and are accessible
        to handlers via ctx.app.config['sys'].

        Usage::

            acies = AciesApp('mic', 'placeholder')
            cli = acies.cli()

            @cli
            @click.option('--threshold', default=0.5)
            def main(**kwargs):
                acies.state.data.update(kwargs)
                acies.run()

            if __name__ == '__main__':
                main()
        """

        def configure(values: dict[str, Any]) -> None:
            deep_merge(self._app_state.config, values)

        return create_acies_cli(configure, **kwargs)

    # ----------------------------- Lifecyle hooks -----------------------------

    def on_startup(self, fn: Callable[..., None]) -> Callable[..., None]:
        self._startup_hooks.append(fn)
        return fn

    def on_shutdown(self, fn: Callable[..., None]) -> Callable[..., None]:
        self._shutdown_hooks.append(fn)
        return fn

    # ---------------------------- task decorators ----------------------------

    def thread(self, fn: Callable[..., None]) -> Callable[..., None]:
        """Long-running thread: fn(ctx, stop) loops until stop.is_set().

        The thread is started after startup hooks and joined (timeout=5s)
        before shutdown hooks run. Any unhandled exception is logged and
        triggers global shutdown. Signature::

            @app.thread
            def my_thread(ctx: AciesContext, stop: threading.Event) -> None:
                while not stop.is_set():
                    ...
        """

        def _run(ctx: AciesContext, stop: threading.Event) -> None:
            try:
                fn(ctx, stop)
            except Exception:
                logger.exception('managed thread %r crashed; triggering shutdown', fn.__name__)
                self.stop()

        self._tasks.append(ThreadSpec(name=fn.__name__, fn=_run))
        return fn

    def subscribe(self, *topics: TopicArg) -> Callable[..., Callable[..., None]]:
        """Message-driven: handler is called for each message on any of the topics.

        The handler must have a parameter named ``msg``::

            @app.subscribe('sensor/data')
            def on_data(ctx: AciesContext, msg: MyMsg) -> None:
                ...
        """

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            if 'msg' not in inspect.signature(fn).parameters:
                raise TypeError(f"subscriber '{fn.__name__}': handler must have a 'msg' parameter")
            hints = get_type_hints(fn)
            msg_type = hints.get('msg')
            self._tasks.append(SubscriberSpec(name=fn.__name__, fn=fn, topics=topics, msg_type=msg_type))
            return fn

        return decorator

    def schedule(self, interval: float) -> Callable[..., Callable[..., None]]:
        """Timer-driven: handler is called every `interval` seconds."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._tasks.append(ScheduleSpec(name=fn.__name__, fn=fn, interval=interval))
            return fn

        return decorator

    def service(self, topic: TopicArg) -> Callable[..., Callable[..., None]]:
        """RPC/queryable: handler is called on queries to topic; return value is the reply.

        The handler must have a ``msg`` parameter with a specific type annotation
        (not bare ``msgspec.Struct``) and a typed return annotation. Both are
        required by ``ctl/schema`` for service discovery::

            @app.service('svc/compute')
            def compute(ctx: AciesContext, msg: MyRequest) -> MyResponse:
                ...
        """

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            if 'msg' not in inspect.signature(fn).parameters:
                raise TypeError(f"service '{fn.__name__}': handler must have a 'msg' parameter")
            hints = get_type_hints(fn)
            msg_type = hints.get('msg')
            return_type = hints.get('return')
            if msg_type is None or msg_type is msgspec.Struct:
                raise TypeError(
                    f"service '{fn.__name__}': 'msg' parameter must have a specific type annotation "
                    f'(not bare msgspec.Struct)'
                )
            if return_type is None:
                raise TypeError(f"service '{fn.__name__}': handler must have a return type annotation")
            self._tasks.append(
                ServiceSpec(name=fn.__name__, fn=fn, topic=topic, msg_type=msg_type, return_type=return_type)
            )
            return fn

        return decorator

    # -------------------------------- Dispatch --------------------------------

    def dispatch(self, job: Job) -> None:
        """Decode, execute, and reply. Called by the Executor on a worker thread.

        This is the only place in the system where msgpack decoding and
        encoding happen — keeping the router and executor byte-agnostic.
        """
        ctx = self._task_ctxs[job.spec]
        match job.spec:
            case ScheduleSpec():
                job.spec.fn(ctx)
            case SubscriberSpec() | ServiceSpec() as spec:
                assert job.raw is not None, 'SubscriberSpec/ServiceSpec job must have raw bytes'
                msg = (  # pyright: ignore[reportUnknownVariableType]
                    msgspec.msgpack.decode(job.raw, type=spec.msg_type)
                    if spec.msg_type is not None
                    else msgspec.msgpack.decode(job.raw)
                )
                result = spec.fn(ctx, msg)
                if job.reply_fn is not None:
                    job.reply_fn(msgspec.msgpack.encode(result))
            case ThreadSpec():
                raise AssertionError(f'ThreadSpec {job.spec.name!r} must never be dispatched')

    # ------------------------------- Run & Stop -------------------------------

    def _resolve_topic(self, topic: TopicArg) -> str:
        """Resolve a topic argument to a concrete string at run() time.

        Handles four forms:
        - ``str`` — returned as-is, or resolved via ``format_map`` if it
          contains ``{placeholders}``.
        - ``Topic`` — resolved via ``ns.topic(*parts, prefix=...)``.
        - ``CtlTopic`` — resolved via ``ns.ctl(*parts)``.
        - ``str`` with ``{key}`` — resolved via ``format_map`` from ``app.state.config``.
        """
        match topic:
            case str():
                if '{' not in topic:
                    return topic
                try:
                    return topic.format_map(self._app_state.config)
                except KeyError as e:
                    raise ValueError(f'topic template {topic!r} references unknown config key {e}') from e
            case Topic():
                if topic.prefix is True:
                    return f'{self._ns.host}/{self._ns.name}/{topic.path}'
                elif topic.prefix:
                    return f'{topic.prefix}/{topic.path}'
                else:
                    return topic.path
            case CtlTopic():
                return f'{self._ns.ctl.base}/{topic.path}'

    def run(self) -> None:  # noqa: C901
        """Start all subsystems, run lifecycle hooks, block until stop() is called.

        Must be called from the main thread. Blocking until shutdown is the
        intended usage — call this as the last statement in main().
        """

        self._ns = Namespace(self._app_state.config['sys']['host'], self._app_state.config['sys']['name'])

        def _make_publish(spec: TaskSpec) -> Callable[[str, bytes], None]:
            def _publish(topic: str, raw: bytes) -> None:
                resolved = self._router.resolve_output(spec, topic)
                if resolved is None:
                    return  # suppressed
                self._router.record_output(spec, resolved)
                self._router.publish(resolved, raw)

            return _publish

        self._task_ctxs = {
            task: AciesContext(
                publish_fn=_make_publish(task),
                query_fn=self._router.query,
                now_fn=time.time_ns,
                app=self._app_state,
                task=TaskState(),
                ns=self._ns,
            )
            for task in self._tasks
        }
        self._executor.start(self.dispatch)
        self._router.start(self._executor)

        # Populate sys.schemas before registering tasks so ctl/schema is
        # ready as soon as the app is wired.
        schemas: dict[str, Any] = {}
        for task in self._tasks:
            if isinstance(task, ServiceSpec):
                entry: dict[str, Any] = {'name': task.name, 'topic': self._resolve_topic(task.topic)}
                if task.msg_type is not None and task.msg_type is not msgspec.Struct:
                    req: dict[str, Any] = {'schema': msgspec.json.schema(task.msg_type)}
                    enc = _get_msg_encoding_metadata(task.msg_type)
                    if enc is not None:
                        req['encoding'] = enc
                    entry['request'] = req
                if task.return_type is not None and task.return_type is not type(None):
                    resp: dict[str, Any] = {'schema': msgspec.json.schema(task.return_type)}
                    enc = _get_msg_encoding_metadata(task.return_type)
                    if enc is not None:
                        resp['encoding'] = enc
                    entry['response'] = resp
                schemas[task.id] = entry
        self._app_state.config['sys']['schemas'] = schemas

        # Register tasks before startup hooks so the app is fully wired
        # when user code in on_startup runs.
        for task in self._tasks:
            match task:
                case SubscriberSpec():
                    for topic in task.topics:
                        self._router.subscribe(self._resolve_topic(topic), task)
                case ServiceSpec():
                    self._router.advertise(self._resolve_topic(task.topic), task)
                case ScheduleSpec() | ThreadSpec():
                    pass  # handled by timer thread

        periodic_tasks = [t for t in self._tasks if isinstance(t, ScheduleSpec)]
        if periodic_tasks:
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                args=(periodic_tasks,),
                name='timer',
                daemon=True,
            )
            self._timer_thread.start()

        hook_ctx = AciesContext(
            publish_fn=self._router.publish,
            query_fn=self._router.query,
            now_fn=time.time_ns,
            app=self._app_state,
            task=TaskState(),
            ns=self._ns,
        )

        with self._app_state.lock:
            self._app_state.config['sys']['state'] = 'active'

        startup_ok = False
        managed_threads: list[threading.Thread] = []
        with temporary_signal_handlers(self.stop):
            try:
                for hook in self._startup_hooks:
                    hook(hook_ctx)
                startup_ok = True

                for spec in (t for t in self._tasks if isinstance(t, ThreadSpec)):
                    t = threading.Thread(
                        target=spec.fn,
                        args=(self._task_ctxs[spec], self._stop_event),
                        name=spec.name,
                        daemon=True,
                    )
                    t.start()
                    managed_threads.append(t)

                # Block until stop() is called (via signal, or directly by app code).
                # SIGINT/SIGTERM are handled by temporary_signal_handlers above,
                # which calls stop() -> sets the event -> wait() returns normally.
                _ = self._stop_event.wait()

            finally:
                self._router.stop()
                self._executor.stop()
                deadline = time.monotonic() + 5.0
                for t in managed_threads:
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        t.join(timeout=remaining)
                if startup_ok:
                    for hook in self._shutdown_hooks:
                        hook(hook_ctx)

    def stop(self) -> None:
        """Signal run() to begin shutdown. Safe to call from any thread."""
        self._stop_event.set()

    def _timer_loop(self, schedule_specs: list[ScheduleSpec]) -> None:
        """Fire ScheduleSpec jobs at their configured intervals.

        Uses a min-heap of (next_fire_time, spec) so a single thread handles
        all timers. Sleeps exactly until the next due time via
        _stop_event.wait(timeout), which also serves as the shutdown signal.
        """
        now = time.monotonic()
        heap: list[tuple[float, ScheduleSpec]] = [(now + spec.interval, spec) for spec in schedule_specs]
        heapq.heapify(heap)

        while True:
            next_fire, spec = heap[0]
            delay = next_fire - time.monotonic()
            if delay > 0 and self._stop_event.wait(timeout=delay):
                break  # shutdown signalled during sleep
            if self._stop_event.is_set():
                break
            _ = heapq.heapreplace(heap, (time.monotonic() + spec.interval, spec))
            self._executor.enqueue(Job(spec=spec, raw=None))
