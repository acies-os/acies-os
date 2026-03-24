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
"""

from __future__ import annotations

import heapq
import socket
import threading
import time
import uuid
from typing import Any, Callable, get_type_hints

import msgspec

from ._cli import create_acies_cli
from .context import AciesContext, AppState, TaskState, deep_merge
from .executor import Executor
from .namespace import Namespace
from .router import Router
from .task import Job, ScheduleSpec, ServiceSpec, SubscriberSpec, TaskSpec
from .transport import ZenohTransport


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
        self._app_state.config['sys'] = {'host': resolved_host, 'name': resolved_name}
        self._ns: Namespace = Namespace(resolved_host, resolved_name)

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

    def subscribe(self, *topics: str) -> Callable[..., Callable[..., None]]:
        """Message-driven: handler is called for each message on any of the topics."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
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

    def service(self, topic: str) -> Callable[..., Callable[..., None]]:
        """RPC/queryable: handler is called on queries to topic; return value is the reply."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            hints = get_type_hints(fn)
            msg_type = hints.get('msg')
            self._tasks.append(ServiceSpec(name=fn.__name__, fn=fn, topic=topic, msg_type=msg_type))
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

    # ------------------------------- Run & Stop -------------------------------

    def run(self) -> None:
        """Start all subsystems, run lifecycle hooks, block until stop() is called."""
        self._task_ctxs = {
            task: AciesContext(
                publish_fn=self._router.publish,
                query_fn=self._router.query,
                app=self._app_state,
                task=TaskState(),
                ns=self._ns,
            )
            for task in self._tasks
        }
        self._executor.start(self.dispatch)
        self._router.start(self._executor)

        # Register tasks before startup hooks so the app is fully wired
        # when user code in on_startup runs.
        for task in self._tasks:
            match task:
                case SubscriberSpec():
                    for topic in task.topics:
                        self._router.subscribe(topic, task)
                case ServiceSpec():
                    self._router.advertise(task.topic, task)
                case ScheduleSpec():
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
            app=self._app_state,
            task=TaskState(),
            ns=self._ns,
        )

        for hook in self._startup_hooks:
            hook(hook_ctx)

        _ = self._stop_event.wait()

        self._router.stop()
        self._executor.stop()

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
