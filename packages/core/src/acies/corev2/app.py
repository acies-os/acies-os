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
from .task import ScheduleSpec, ServiceSpec, SubscriberSpec


class AciesApp:
    def __init__(self, name: str, router: Router | None = None) -> None:
        self._name: str = name
        self._router: Router = router if router is not None else Router()
        self._executor: Executor = Executor()
        self._specs: list[SubscriberSpec | ScheduleSpec | ServiceSpec] = []
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
    # Task decorators
    # ------------------------------------------------------------------

    def subscribe(self, *topics: str) -> Callable[..., Callable[..., None]]:
        """Message-driven: handler is called for each message on any of the topics."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._specs.append(SubscriberSpec(name=fn.__name__, fn=fn, topics=topics))
            return fn

        return decorator

    def schedule(self, interval: float) -> Callable[..., Callable[..., None]]:
        """Timer-driven: handler is called every `interval` seconds."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._specs.append(ScheduleSpec(name=fn.__name__, fn=fn, interval=interval))
            return fn

        return decorator

    def service(self, topic: str) -> Callable[..., Callable[..., None]]:
        """RPC/queryable: handler is called on queries to topic; return value is the reply."""

        def decorator(fn: Callable[..., None]) -> Callable[..., None]:
            self._specs.append(ServiceSpec(name=fn.__name__, fn=fn, topics=(topic,)))
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all subsystems, run lifecycle hooks, block until stop() is called."""
        # TODO: Phase 2 —
        #   1. Build one AciesContext per spec (reused across all jobs of that spec)
        #   2. executor.start(dispatch)
        #   3. router.start(executor)
        #   4. call startup hooks
        #   5. router.subscribe / router.advertise for each spec
        #   6. start timer thread (_timer_loop)
        #   7. self._stop_event.wait()
        #   8. router.stop(), executor.stop()
        #   9. call shutdown hooks
        ...

    def stop(self) -> None:
        """Signal run() to begin shutdown. Safe to call from any thread."""
        self._stop_event.set()

    def _timer_loop(self) -> None:
        """Single shared timer thread. Manages all ScheduleSpecs via a
        priority queue of (next_fire_time, spec). Wakes exactly when the
        next timer is due."""
        # TODO: Phase 2
        ...
