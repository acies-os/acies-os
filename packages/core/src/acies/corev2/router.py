"""Router — inbound queue, router thread, topic-to-Job dispatch.

The Router sits between the transport layer and the executor:
  - Transport receiver threads push (topic, msg) into the inbound queue.
  - The router thread drains the queue, matches topics to TaskSpecs,
    creates Jobs, and calls executor.enqueue().
  - Outbound messages (publish, query) are synchronous — called directly
    from worker threads via AciesContext, no router thread involved.

Routing is prefix-based: for each outbound call, the router asks each
registered transport can_handle(topic) and uses the first match. Raises
RuntimeError if no transport can handle the topic.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

from .task import TaskSpec
from .transport import Transport

if TYPE_CHECKING:
    from .executor import Executor
    from .msg import AciesMsg


class Router:
    def __init__(self) -> None:
        self._transports: list[Transport] = []
        self._inbound: queue.Queue[tuple[str, 'AciesMsg', object]] = queue.Queue()
        self._subscriptions: dict[str, list[TaskSpec]] = {}
        self._services: dict[str, TaskSpec] = {}
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()

    def add(self, transport: Transport) -> None:
        """Register a transport backend. Order matters for routing: first
        matching transport wins, so add specific (prefixed) transports before
        catch-all transports like LocalTransport."""
        self._transports.append(transport)

    def start(self, executor: 'Executor') -> None:
        """Start all transports and the router thread."""
        # TODO: Phase 2 — start each transport with on_message callback,
        # start self._thread running _route_loop(executor)
        ...

    def stop(self) -> None:
        """Stop router thread and all transports."""
        # TODO: Phase 2
        ...

    def subscribe(self, topic: str, spec: TaskSpec) -> None:
        """Register a TaskSpec to receive messages on topic."""
        self._subscriptions.setdefault(topic, []).append(spec)
        # TODO: Phase 2 — also call transport.subscribe(topic) on matching transports

    def advertise(self, topic: str, spec: TaskSpec) -> None:
        """Register a TaskSpec as a queryable service on topic."""
        self._services[topic] = spec
        # TODO: Phase 2 — call transport.advertise(topic, reply_fn_factory)

    def publish(self, topic: str, msg: 'AciesMsg') -> None:
        """Send a message. Called synchronously from worker threads."""
        transport = self._transport_for(topic)
        transport.publish(topic, msg)

    def query(self, topic: str, msg: 'AciesMsg', timeout: float) -> 'AciesMsg | None':
        """Synchronous RPC. Called from worker threads via AciesContext."""
        transport = self._transport_for(topic)
        return transport.query(topic, msg, timeout)

    def _transport_for(self, topic: str) -> Transport:
        for t in self._transports:
            if t.can_handle(topic):
                return t
        raise RuntimeError(f'No transport can handle topic: {topic!r}')

    def _route_loop(self, executor: 'Executor') -> None:
        # TODO: Phase 2 — drain self._inbound, match topic → specs,
        # create Job(spec, msg) or Job(spec, msg, reply_fn), call executor.enqueue()
        ...
