"""System I/O graph built from per-node ``ctl/io`` snapshots.

Pure functions only (no I/O, no app state), so the logic can be tested
without a running gateway.

The graph is bipartite: *task* nodes and *topic* nodes.

- task -> topic: the task has published to the topic (an output).
- topic -> task: the task subscribes to, or serves, the topic (an input).
- topic -> topic: a concrete output topic matches a wildcard input topic
  (e.g. ``rs1/vfm/vehicle`` -> ``**/vehicle``).

Edges come from *observed* outputs, so a task that has not published yet has
no outgoing edges, and a topic that was remapped away stays until restart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from acies.core.msg import AciesIoResponse
from acies.core.namespace import matches

# tasks every AciesApp registers (see acies.core._control); hidden by default
SYSTEM_TASKS = frozenset({'_heartbeat', '_kv', '_route', '_io', '_schema'})


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def build_io_graph(io_cache: Mapping[str, AciesIoResponse], *, include_system: bool = False) -> dict[str, Any]:
    """Build the I/O graph from ``{app source: AciesIoResponse}``.

    Returns ``{'nodes': [...], 'edges': [[src_id, dst_id], ...]}``, sorted so
    the result is deterministic. Node shapes::

        {'id': 'task:<spec id>', 'kind': 'task', 'app': 'rs1/mic', 'name': 'mic_loop', 'timestamp': <ns>}
        {'id': 'topic:<topic>', 'kind': 'topic', 'topic': 'rs1/mic'}

    ``timestamp`` on a task is when its app produced the snapshot, so the
    caller can tell stale data apart. Topic nodes exist for every topic that
    is an input or an output of a listed task, so unmatched inputs and
    unconsumed outputs stay visible.
    """
    nodes: dict[str, dict[str, Any]] = {}
    edges: set[tuple[str, str]] = set()
    inputs: set[str] = set()
    outputs: set[str] = set()

    def topic_id(topic: str) -> str:
        node_id = f'topic:{topic}'
        nodes.setdefault(node_id, {'id': node_id, 'kind': 'topic', 'topic': topic})
        return node_id

    for source, resp in io_cache.items():
        for spec_id, entry in resp.io.items():
            name = str(entry['name'])
            if name in SYSTEM_TASKS and not include_system:
                continue
            task = f'task:{spec_id}'
            nodes[task] = {'id': task, 'kind': 'task', 'app': source, 'name': name, 'timestamp': resp.timestamp}
            for topic in _as_list(entry['inputs']):
                edges.add((topic_id(topic), task))
                inputs.add(topic)
            for topic in _as_list(entry['outputs']):
                edges.add((task, topic_id(topic)))
                outputs.add(topic)

    # wildcard inputs: link each concrete output topic that they match
    for pattern in inputs:
        if '*' not in pattern:
            continue
        for topic in outputs:
            if matches(pattern, topic):
                edges.add((topic_id(topic), topic_id(pattern)))

    return {
        'nodes': sorted(nodes.values(), key=lambda n: n['id']),
        'edges': [list(e) for e in sorted(edges)],
    }
