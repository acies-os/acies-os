"""Per-node stats built from ``sys/memory`` and ``sys/timing`` snapshots.

Pure functions only (no I/O, no app state), so the logic can be tested
without a running gateway.

Each classifier node keeps its latest process-memory sample in ``sys/memory``
(bytes) and its rolling inference timings in ``sys/timing``. The gateway
collects both into a cache of ``{source: {'memory', 'timing', 'queried_at'}}``;
this module turns that cache into the payload published on ``ws://stats``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_NS_PER_S = 1_000_000_000

# model services by deployment style; a service name is the last segment of a node's source
FM_SERVICES = frozenset({'vfm_encoder', 'vfm_vehicle', 'vfm_drone'})
E2E_SERVICES = frozenset({'e2e_vehicle', 'e2e_drone'})
GROUPS = ('fm', 'e2e', 'other')

# resident-memory fields in order of preference: pss is additive across processes (Linux only),
# uss is the closest substitute elsewhere, rss counts shared library pages in every process
_BASIS = ('pss', 'uss', 'rss')


def group_of(name: str) -> str:
    """Deployment style of a service: 'fm' (encoder + heads), 'e2e' (end-to-end) or 'other'."""
    if name in FM_SERVICES:
        return 'fm'
    if name in E2E_SERVICES:
        return 'e2e'
    return 'other'


def resident(memory: Mapping[str, Any]) -> tuple[str, int] | None:
    """Return (basis, bytes) using the best available resident-memory field, or None."""
    for basis in _BASIS:
        value = memory.get(basis)
        if isinstance(value, int):
            return basis, value
    return None


def build_stats(stats_cache: Mapping[str, Mapping[str, Any]], now_ns: int) -> dict[str, Any]:
    """Build the ``ws://stats`` payload from ``{source: {'memory', 'timing', 'queried_at'}}``.

    Returns ``{'timestamp', 'nodes', 'hosts'}``. Node shape::

        {'host': 'rs1', 'name': 'vfm_encoder', 'group': 'fm', 'memory': {...}|None, 'timing': {...}|None,
         'queried_at': <ns>, 'age_s': <seconds since the gateway last got an answer>}

    ``memory`` and ``timing`` are passed through as the node reported them: ``sys/memory`` (bytes, with
    ``tag`` and ``time_ns``) and ``sys/timing`` (``{'time_ns', 'ops'}``). Per host,
    ``resident`` and ``weights`` sum the nodes of each group ('fm', 'e2e', 'other', plus 'total') in
    bytes; ``basis`` says which field ``resident`` was summed from ('pss', 'uss', 'rss', or 'mixed').
    ``weights`` is the model parameter/buffer bytes only, ``resident`` the whole process.
    """
    nodes: dict[str, dict[str, Any]] = {}
    hosts: dict[str, dict[str, Any]] = {}
    bases: dict[str, set[str]] = {}

    for source in sorted(stats_cache):
        entry = stats_cache[source]
        host, _, rest = source.partition('/')
        name = rest.rsplit('/', 1)[-1]
        group = group_of(name)
        memory: Mapping[str, Any] | None = entry.get('memory')
        queried_at: int = entry['queried_at']
        nodes[source] = {
            'host': host,
            'name': name,
            'group': group,
            'memory': memory,
            'timing': entry.get('timing'),
            'queried_at': queried_at,
            'age_s': round((now_ns - queried_at) / _NS_PER_S, 1),
        }

        if memory is None:
            continue
        info = hosts.setdefault(
            host,
            {
                'basis': None,
                'resident': dict.fromkeys((*GROUPS, 'total'), 0),
                'weights': dict.fromkeys((*GROUPS, 'total'), 0),
            },
        )
        res = resident(memory)
        if res is not None:
            basis, nbytes = res
            bases.setdefault(host, set()).add(basis)
            info['resident'][group] += nbytes
            info['resident']['total'] += nbytes
        weights = memory.get('weights')
        if isinstance(weights, int):
            info['weights'][group] += weights
            info['weights']['total'] += weights

    for host, info in hosts.items():
        host_bases = bases.get(host, set())
        info['basis'] = next(iter(host_bases)) if len(host_bases) == 1 else ('mixed' if host_bases else None)

    return {'timestamp': now_ns, 'nodes': nodes, 'hosts': hosts}
