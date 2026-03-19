"""AciesApp — user-facing entry point.

Owns the Router, Executor, timer thread, and TaskSpec registry.
Decorators register TaskSpecs; run() wires everything together and blocks.

Thread model (implemented in Phase 2):
  Main thread      — runs run(); blocks on _stop_event after wiring
  Transport threads — owned by each Transport; push to router inbound queue
  Router thread    — drains inbound queue; creates Jobs; calls executor.enqueue()
  Timer thread     — one shared thread; fires SCHEDULE jobs into executor
  Dispatcher thread — owned by Executor; drains internal queue; submits to pool
  Worker threads   — execute handlers; call ctx.publish() synchronously
"""

from __future__ import annotations

import threading
from typing import Callable

from .context import AciesContext
from .executor import Executor
from .router import Router
from .task import TaskKind, TaskSpec


class AciesApp:
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._router: Router = Router()
        self._executor: Executor = Executor()
        # AciesContext receives router capabilities as plain callables —
        # no direct import of Router in context.py, breaking the import cycle.
        self._ctx: AciesContext = AciesContext(
            publish_fn=self._router.publish,
            query_fn=self._router.query,
        )
        self._specs: list[TaskSpec] = []
        self._startup_hooks: list[Callable[..., None]] = []
        self._shutdown_hooks: list[Callable[..., None]] = []
        self._stop_event: threading.Event = threading.Event()
        self._timer_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_startup(self, fn: Callable[..., None]) -> Callable[..., None]:
        self._startup_hooks.append(fn)
        return fn

    def on_shutdown(self, fn: Callable[..., None]) -> Callable[..., None]:
        self._shutdown_hooks.append(fn)
        return fn

    # ------------------------------------------------------------------
    # Task decorators — register TaskSpecs by trigger mechanism
    # ------------------------------------------------------------------

    def subscribe(self, *topics: str) -> Callable[..., Callable[..., None]]:
        """Message-driven: handler is called for each message on any of the topics."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._specs.append(TaskSpec(name=fn.__name__, kind=TaskKind.SUBSCRIBE, fn=fn, topics=topics))
            return fn

        return decorator

    def schedule(self, interval: float) -> Callable[..., Callable[..., None]]:
        """Timer-driven: handler is called every `interval` seconds."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._specs.append(TaskSpec(name=fn.__name__, kind=TaskKind.SCHEDULE, fn=fn, interval=interval))
            return fn

        return decorator

    def produce(self, fn: Callable[..., None]) -> Callable[..., Callable[..., None]]:
        """External event-driven: wraps fn so it receives AciesContext.
        The decorated function is called by user code or an external SDK callback."""
        # FIXME: this is not quit right. Think a thread getting data from microphone driver, and send to a topic.
        self._specs.append(TaskSpec(name=fn.__name__, kind=TaskKind.PRODUCE, fn=fn))
        return fn

    def service(self, topic: str) -> Callable[..., Callable[..., None]]:
        """RPC/queryable: handler is called on queries to topic; return value is the reply."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._specs.append(TaskSpec(name=fn.__name__, kind=TaskKind.SERVICE, fn=fn, topics=(topic,)))
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all subsystems, run lifecycle hooks, block until stop() is called."""
        # TODO: Phase 2 —
        #   1. executor.start(dispatch=lambda job: job.spec.fn(self._ctx, job.msg))
        #   2. router.start(executor)
        #   3. call startup hooks
        #   4. router.subscribe / router.advertise for each spec
        #   5. start timer thread (_timer_loop)
        #   6. self._stop_event.wait()
        #   7. router.stop(), executor.stop()
        #   8. call shutdown hooks
        ...

    def stop(self) -> None:
        """Signal run() to begin shutdown. Safe to call from any thread."""
        self._stop_event.set()

    def _timer_loop(self) -> None:
        """Single shared timer thread. Manages all SCHEDULE specs via a
        priority queue of (next_fire_time, spec). Blocks on executor.enqueue()
        with a dynamic timeout so it wakes exactly when the next timer is due."""
        # TODO: Phase 2
        ...

    def _specs_of(self, kind: TaskKind) -> list[TaskSpec]:
        return [s for s in self._specs if s.kind == kind]
