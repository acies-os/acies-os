"""Router — inbound queue, router thread, topic-to-Job dispatch.

The Router sits between the transport layer and the executor:
  - Transport receiver threads push (topic, raw, reply_fn) into the inbound queue.
  - The router thread drains the queue, matches topics to TaskSpecs,
    creates Jobs, and calls executor.enqueue().
  - Outbound messages (publish, query) are synchronous — called directly
    from worker threads via AciesContext, no router thread involved.

Transport selection
-------------------
The router uses a default transport plus optional prefix overrides:

  ws://...   →  WebSocketTransport (registered via prefix='ws://')
  anything   →  default transport

In production the default is ZenohTransport (one session per process,
connected via unix domain socket to the local zenohd). Locality between
processes — whether on the same device or across devices — is handled
transparently by the zenoh network; the router has no topology awareness.

In tests the default is LocalTransport, which requires no daemon.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any

import msgspec

from .task import TaskSpec
from .transport import Transport

if TYPE_CHECKING:
    from .executor import Executor

_SENTINEL = object()


class Router:
    def __init__(self) -> None:
        self._default_transport: Transport | None = None
        # Explicit prefix routes, e.g. ('ws://', ws_transport). First match wins.
        self._prefix_routes: list[tuple[str, Transport]] = []

        self._inbound: queue.Queue[tuple[str, bytes, Any]] = queue.Queue()
        self._subscriptions: dict[str, list[TaskSpec]] = {}
        self._services: dict[str, TaskSpec] = {}
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()

    def add_transport(
        self,
        transport: Transport,
        *,
        prefix: str | None = None,
    ) -> None:
        """Register a transport.

        Without prefix: becomes the default transport (ZenohTransport in
        production, LocalTransport in tests). Only one default may be set;
        the last call wins.

        With prefix: routes topics starting with that string to the given
        transport (e.g. prefix='ws://' for WebSocketTransport). Prefix routes
        are checked before the default, in insertion order.
        """
        if prefix is not None:
            self._prefix_routes.append((prefix, transport))
        else:
            self._default_transport = transport

    def start(self, executor: 'Executor') -> None:
        """Start all transports and the router thread."""
        # TODO: Phase 2 — start each transport with _on_message callback,
        # start self._thread running _route_loop(executor)
        ...

    def stop(self) -> None:
        """Stop router thread and all transports."""
        # TODO: Phase 2
        ...

    def subscribe(self, topic: str, spec: TaskSpec) -> None:
        """Register a TaskSpec to receive messages on topic."""
        self._subscriptions.setdefault(topic, []).append(spec)
        # TODO: Phase 2 — call transport.subscribe(topic) on the matching transport

    def advertise(self, topic: str, spec: TaskSpec) -> None:
        """Register a TaskSpec as a queryable service on topic."""
        self._services[topic] = spec
        # TODO: Phase 2 — call transport.advertise(topic, reply_fn_factory)

    def publish(self, topic: str, msg: msgspec.Struct) -> None:
        """Encode msg and send. Called synchronously from worker threads via AciesContext."""
        transport = self._transport_for(topic)
        transport.publish(topic, msgspec.msgpack.encode(msg))

    def query(self, topic: str, msg: msgspec.Struct, timeout: float) -> msgspec.Struct | None:
        """Synchronous RPC. Encodes request, sends, decodes reply.
        Called from worker threads via AciesContext."""
        transport = self._transport_for(topic)
        raw = transport.query(topic, msgspec.msgpack.encode(msg), timeout)
        if raw is None:
            return None
        # TODO: Phase 2 — decode with the reply type once reply typing is defined
        return msgspec.msgpack.decode(raw)

    def _on_message(self, topic: str, raw: bytes, reply_fn: Any = None) -> None:
        """Transport callback — push into the inbound queue."""
        self._inbound.put((topic, raw, reply_fn))

    def _transport_for(self, topic: str) -> Transport:
        for prefix, transport in self._prefix_routes:
            if topic.startswith(prefix):
                return transport
        if self._default_transport is not None:
            return self._default_transport
        raise RuntimeError(f'No transport for topic: {topic!r}')

    def _route_loop(self, executor: 'Executor') -> None:
        # TODO: Phase 2 — drain self._inbound; skip acies/ctrl/* (handle internally);
        # decode msgspec.msgpack.decode(raw, type=spec.msg_type) for each matching spec;
        # create Job(spec, decoded_msg) or Job(spec, decoded_msg, reply_fn);
        # call executor.enqueue(job)
        ...
