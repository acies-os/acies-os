"""Tests for the pure stats aggregation used by the gateway."""

from typing import Any

from acies.gateway.stats import build_stats, group_of, resident

_MIB = 1024 * 1024
_NOW = 100 * 1_000_000_000


def _entry(
    memory: dict[str, Any] | None, timing: dict[str, Any] | None = None, queried_at: int = _NOW
) -> dict[str, Any]:
    return {'memory': memory, 'timing': timing, 'queried_at': queried_at}


def test_empty_cache():
    assert build_stats({}, _NOW) == {'timestamp': _NOW, 'nodes': {}, 'hosts': {}}


def test_group_of():
    assert group_of('vfm_encoder') == 'fm'
    assert group_of('vfm_vehicle') == 'fm'
    assert group_of('vfm_drone') == 'fm'
    assert group_of('e2e_vehicle') == 'e2e'
    assert group_of('e2e_drone') == 'e2e'
    assert group_of('spar') == 'other'
    assert group_of('mic') == 'other'


def test_resident_prefers_pss_then_uss_then_rss():
    assert resident({'rss': 3, 'uss': 2, 'pss': 1}) == ('pss', 1)
    assert resident({'rss': 3, 'uss': 2}) == ('uss', 2)
    assert resident({'rss': 3}) == ('rss', 3)
    assert resident({'weights': 5}) is None


def test_node_fields_and_age():
    mem = {'rss': 10, 'weights': 4}
    timing = {'time_ns': _NOW, 'ops': {'infer': {'count': 3}}}
    stats = build_stats({'rs1/vfm_encoder': _entry(mem, timing, queried_at=_NOW - 12_340_000_000)}, _NOW)
    node = stats['nodes']['rs1/vfm_encoder']
    assert node['host'] == 'rs1'
    assert node['name'] == 'vfm_encoder'
    assert node['group'] == 'fm'
    assert node['memory'] == mem
    assert node['timing'] == timing
    assert node['age_s'] == 12.3


def test_host_sums_by_group_using_pss():
    cache = {
        'rs1/vfm_encoder': _entry({'pss': 200 * _MIB, 'uss': 190 * _MIB, 'weights': 45 * _MIB}),
        'rs1/vfm_vehicle': _entry({'pss': 120 * _MIB, 'uss': 110 * _MIB, 'weights': 4096}),
        'rs1/vfm_drone': _entry({'pss': 120 * _MIB, 'uss': 110 * _MIB, 'weights': 4096}),
        'rs1/e2e_vehicle': _entry({'pss': 210 * _MIB, 'uss': 200 * _MIB, 'weights': 45 * _MIB}),
    }
    host = build_stats(cache, _NOW)['hosts']['rs1']
    assert host['basis'] == 'pss'
    assert host['resident'] == {'fm': 440 * _MIB, 'e2e': 210 * _MIB, 'other': 0, 'total': 650 * _MIB}
    assert host['weights'] == {'fm': 45 * _MIB + 8192, 'e2e': 45 * _MIB, 'other': 0, 'total': 90 * _MIB + 8192}


def test_hosts_are_separate_and_basis_follows_platform():
    cache = {
        'rs1/vfm_encoder': _entry({'pss': 5, 'uss': 4, 'rss': 6}),
        'mac/vfm_encoder': _entry({'uss': 7, 'rss': 9}),
    }
    hosts = build_stats(cache, _NOW)['hosts']
    assert hosts['rs1']['basis'] == 'pss'
    assert hosts['rs1']['resident']['total'] == 5
    assert hosts['mac']['basis'] == 'uss'
    assert hosts['mac']['resident']['total'] == 7


def test_mixed_basis_on_one_host_is_flagged():
    cache = {
        'rs1/vfm_encoder': _entry({'pss': 5}),
        'rs1/vfm_vehicle': _entry({'uss': 3}),
    }
    assert build_stats(cache, _NOW)['hosts']['rs1']['basis'] == 'mixed'


def test_timing_only_node_is_listed_but_not_counted():
    cache = {'rs1/spar': _entry(None, {'time_ns': _NOW, 'ops': {'infer': {'count': 1}}})}
    stats = build_stats(cache, _NOW)
    assert stats['nodes']['rs1/spar']['memory'] is None
    assert stats['hosts'] == {}


def test_memory_without_weights_still_counts_resident():
    host = build_stats({'rs1/spar': _entry({'rss': 8})}, _NOW)['hosts']['rs1']
    assert host['resident'] == {'fm': 0, 'e2e': 0, 'other': 8, 'total': 8}
    assert host['weights']['total'] == 0
