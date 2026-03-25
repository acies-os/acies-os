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

from ._concurrency import SENTINEL, Sentinel
from .msg import TopicRename
from .namespace import matches
from .task import Job, ServiceSpec, SubscriberSpec, TaskSpec
from .transport import ReplyCallback, Transport

if TYPE_CHECKING:
    from .executor import Executor


class Router:
    def __init__(self) -> None:
        self._default_transport: Transport | None = None
        # Explicit prefix routes, e.g. ('ws://', ws_transport). First match wins.
        self._prefix_routes: list[tuple[str, Transport]] = []

        self._inbound: queue.Queue[tuple[str, bytes, ReplyCallback | None] | Sentinel] = queue.Queue()
        self._subscriptions: dict[str, set[SubscriberSpec]] = {}
        self._services: dict[str, ServiceSpec] = {}
        self._routing_lock: threading.Lock = threading.Lock()
        self._thread: threading.Thread | None = None

        # I/O routing table: inputs populated at subscribe/advertise time;
        # outputs accumulated at runtime via record_output.
        self._spec_inputs: dict[TaskSpec, set[str]] = {}
        self._spec_outputs: dict[TaskSpec, set[str]] = {}
        self._io_lock: threading.Lock = threading.Lock()

        # Output remap table: maps spec -> {original_topic -> effective_topic | None}.
        # None means suppress. Inner dicts are immutable (copy-on-write);
        # _routing_lock serializes concurrent writes only.
        self._output_remap: dict[TaskSpec, dict[str, str | None]] = {}

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
        self._inbound.put(SENTINEL)
        if self._thread:
            self._thread.join()
        for transport in self._all_transports():
            transport.stop()

    def subscribe(self, topic: str, spec: SubscriberSpec) -> None:
        """Register a SubscriberSpec to receive messages on topic."""
        with self._routing_lock:
            self._subscriptions.setdefault(topic, set()).add(spec)
            self._spec_inputs.setdefault(spec, set()).add(topic)
        self._transport_for(topic).subscribe(topic)

    def unsubscribe(self, topic: str, spec: SubscriberSpec) -> None:
        """Remove a SubscriberSpec from topic; undeclares transport subscription if no specs remain."""
        with self._routing_lock:
            specs = self._subscriptions.get(topic)
            if specs:
                specs.discard(spec)
                if not specs:
                    del self._subscriptions[topic]
                    self._transport_for(topic).unsubscribe(topic)
            inputs = self._spec_inputs.get(spec)
            if inputs:
                inputs.discard(topic)

    def advertise(self, topic: str, spec: ServiceSpec) -> None:
        """Register a ServiceSpec as a queryable service on topic."""
        with self._routing_lock:
            self._services[topic] = spec
            self._spec_inputs.setdefault(spec, set()).add(topic)
        self._transport_for(topic).advertise(topic)

    def unadvertise(self, topic: str) -> None:
        """Remove a ServiceSpec queryable from topic."""
        with self._routing_lock:
            spec = self._services.pop(topic, None)
            if spec is not None:
                inputs = self._spec_inputs.get(spec)
                if inputs:
                    inputs.discard(topic)
        self._transport_for(topic).unadvertise(topic)

    def find_spec(self, spec_id: str | None, spec_name: str | None) -> TaskSpec | None:
        """Find a spec by id (preferred) or name (fallback). Only searches specs with inputs."""
        with self._routing_lock:
            specs = list(self._spec_inputs.keys())
        if spec_id is not None:
            for spec in specs:
                if spec.id == spec_id:
                    return spec
        if spec_name is not None:
            for spec in specs:
                if spec.name == spec_name:
                    return spec
        return None

    def remap_output(self, spec: TaskSpec, rename: TopicRename) -> None:
        """Update the output remap table for spec.

        old=X, new=Y  — redirect publishes from X to Y.
        old=X, new=None — suppress publishes to X.
        old=None — noop (no original topic to intercept).

        Consecutive renames are collapsed: rename(t1->t2) then rename(t2->t3)
        results in a single effective entry t1->t3.

        Copy-on-write: builds a new inner dict and replaces the reference
        atomically. Inner dicts are never mutated after assignment, so
        resolve_output readers need no lock.
        """
        if rename.old is None:
            return
        with self._routing_lock:
            old_table = self._output_remap.get(spec, {})
            new_table = dict(old_table)
            updated = False
            # collapse conseutive renames
            # .e.g. if old_table has t1->t2 and rename is t2->t3, update to t1->t3
            for src, dst in new_table.items():
                if dst == rename.old:
                    new_table[src] = rename.new
                    updated = True
            if not updated:
                new_table[rename.old] = rename.new
            self._output_remap[spec] = new_table

    def resolve_output(self, spec: TaskSpec, topic: str) -> str | None:
        """Return the effective output topic for spec after applying any remap.

        Returns None if the publish should be suppressed.
        Returns topic unchanged if no remap entry exists for it.

        Lock-free: inner dicts are immutable snapshots (copy-on-write in
        remap_output), so reading them requires no synchronization.
        """
        table = self._output_remap.get(spec)
        if table is None:
            return topic
        return table.get(topic, topic)

    def record_output(self, spec: TaskSpec, topic: str) -> None:
        """Record that spec published to topic. Called from per-spec publish closures.

        Output topics are observed at runtime and accumulate over the life of
        the app — conditional publish paths will appear once they are exercised.
        Thread-safe: may be called concurrently from worker threads.
        """
        outputs = self._spec_outputs.get(spec)
        if outputs is not None and topic in outputs:
            return  # fast path: already recorded, no lock needed
        with self._io_lock:
            self._spec_outputs.setdefault(spec, set()).add(topic)

    @property
    def io_map(self) -> dict[str, dict[str, str | list[str]]]:
        """Snapshot of the I/O routing table keyed by spec UUID.

        Returns ``{spec.id: {'name': ..., 'inputs': [...], 'outputs': [...]}, ...}``.
        Inputs are the concrete topics registered at startup; outputs are all
        topics the task has published to since the app started.
        """
        with self._io_lock:
            specs = set(self._spec_inputs) | set(self._spec_outputs)
            return {
                spec.id: {
                    'name': spec.name,
                    'inputs': sorted(self._spec_inputs.get(spec, set())),
                    'outputs': sorted(self._spec_outputs.get(spec, set())),
                }
                for spec in specs
            }

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

    def _on_message(self, topic: str, raw: bytes, reply_fn: ReplyCallback | None = None) -> None:
        """Transport callback — push into the inbound queue."""
        self._inbound.put((topic, raw, reply_fn))

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
            if item is SENTINEL:
                break
            assert isinstance(item, tuple)
            topic, raw, reply_fn = item

            # TODO: log/warn unmatched messages
            with self._routing_lock:
                if reply_fn is not None:
                    # Incoming query — route to the matching service spec
                    for pattern, spec in self._services.items():
                        if matches(pattern, topic):
                            executor.enqueue(Job(spec=spec, raw=raw, reply_fn=reply_fn))
                            break
                else:
                    # Incoming pub — fan out to all matching subscriber specs
                    for pattern, specs in self._subscriptions.items():
                        if matches(pattern, topic):
                            for spec in specs:
                                executor.enqueue(Job(spec=spec, raw=raw))
