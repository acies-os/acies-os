"""Router — inbound queue, router thread, topic-to-Job dispatch.

The Router connects transport topics to handler specs and vice versa.
It deliberately has no knowledge of message encoding:

  Inbound:  receive raw bytes from transport → match topic to specs
            → create Job(spec, raw) → enqueue in executor
            → dispatch (worker thread) decodes raw and calls the handler

  Outbound: AciesContext.publish encodes struct → bytes → router forwards
            bytes to the right transport

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
from typing import TYPE_CHECKING

from .task import Job, ServiceSpec, SubscriberSpec
from .transport import SendBytes, Transport, _topic_matches

if TYPE_CHECKING:
    from .executor import Executor


class _Sentinel:
    pass


_SENTINEL = _Sentinel()


class Router:
    def __init__(self) -> None:
        self._default_transport: Transport | None = None
        # Explicit prefix routes, e.g. ('ws://', ws_transport). First match wins.
        self._prefix_routes: list[tuple[str, Transport]] = []

        self._inbound: queue.Queue[tuple[str, bytes, SendBytes | None] | _Sentinel] = queue.Queue()
        self._subscriptions: dict[str, list[SubscriberSpec]] = {}
        self._services: dict[str, ServiceSpec] = {}
        self._thread: threading.Thread | None = None

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
        for transport in self._all_transports():
            transport.start(self._on_message)
        self._thread = threading.Thread(target=self._route_loop, args=(executor,), name='router', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop router thread and all transports."""
        self._inbound.put(_SENTINEL)
        if self._thread:
            self._thread.join()
        for transport in self._all_transports():
            transport.stop()

    def subscribe(self, topic: str, spec: SubscriberSpec) -> None:
        """Register a SubscriberSpec to receive messages on topic."""
        self._subscriptions.setdefault(topic, []).append(spec)
        self._transport_for(topic).subscribe(topic)

    def advertise(self, topic: str, spec: ServiceSpec) -> None:
        """Register a ServiceSpec as a queryable service on topic."""
        self._services[topic] = spec
        self._transport_for(topic).advertise(topic)

    def publish(self, topic: str, raw: bytes) -> None:
        """Forward raw bytes to the transport for topic.

        Called from worker threads via AciesContext, which handles encoding.
        """
        self._transport_for(topic).publish(topic, raw)

    def query(self, topic: str, raw: bytes, timeout: float) -> bytes | None:
        """Forward a raw query to the transport; return raw reply bytes or None.

        Called from worker threads via AciesContext, which handles encoding
        and decoding of the request and reply structs.
        """
        return self._transport_for(topic).query(topic, raw, timeout)

    def _on_message(self, topic: str, raw: bytes, send_bytes: SendBytes | None = None) -> None:
        """Transport callback — push into the inbound queue."""
        self._inbound.put((topic, raw, send_bytes))

    def _transport_for(self, topic: str) -> Transport:
        for prefix, transport in self._prefix_routes:
            if topic.startswith(prefix):
                return transport
        if self._default_transport is not None:
            return self._default_transport
        raise RuntimeError(f'No transport for topic: {topic!r}')

    def _all_transports(self) -> list[Transport]:
        seen: set[int] = set()
        result: list[Transport] = []
        for _, t in self._prefix_routes:
            if id(t) not in seen:
                seen.add(id(t))
                result.append(t)
        if self._default_transport is not None and id(self._default_transport) not in seen:
            result.append(self._default_transport)
        return result

    def _route_loop(self, executor: 'Executor') -> None:
        while True:
            item = self._inbound.get()
            if item is _SENTINEL:
                break
            assert isinstance(item, tuple)
            topic, raw, send_bytes = item

            if topic.startswith('acies/ctrl/'):
                # TODO: it should be '**/ctl/**', or do we need this?
                # TODO: should all control be done via RPC? Or both RPC and pub/sub?
                raise NotImplementedError('Control messages handling not implemented yet')

            # TODO: log/warn unmatched messages
            if send_bytes is not None:
                # Incoming query — route to the matching service spec
                for pattern, spec in self._services.items():
                    if _topic_matches(pattern, topic):
                        executor.enqueue(Job(spec=spec, raw=raw, send_bytes=send_bytes))
                        break
            else:
                # Incoming pub — fan out to all matching subscriber specs
                for pattern, specs in self._subscriptions.items():
                    if _topic_matches(pattern, topic):
                        for spec in specs:
                            executor.enqueue(Job(spec=spec, raw=raw))
