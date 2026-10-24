"""Tests for the pure I/O graph builder used by the gateway."""

from typing import Any

from acies.core.msg import AciesIoResponse
from acies.gateway.graph import build_io_graph


def _resp(tasks: dict[str, tuple[str, list[str], list[str]]], timestamp: int = 1) -> AciesIoResponse:
    """tasks: {spec_id: (name, inputs, outputs)}"""
    io: dict[str, dict[str, Any]] = {
        spec_id: {'name': name, 'inputs': ins, 'outputs': outs} for spec_id, (name, ins, outs) in tasks.items()
    }
    return AciesIoResponse(timestamp=timestamp, io=io)


def _edges(graph: dict[str, Any]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in graph['edges']}


def _ids(graph: dict[str, Any], kind: str) -> set[str]:
    return {n['id'] for n in graph['nodes'] if n['kind'] == kind}


def test_empty_cache_gives_empty_graph():
    assert build_io_graph({}) == {'nodes': [], 'edges': []}


def test_publisher_to_subscriber_through_topic():
    cache = {
        'rs1/mic': _resp({'a': ('mic_loop', [], ['rs1/mic/audio'])}),
        'rs1/vfm': _resp({'b': ('classify', ['rs1/mic/audio'], [])}),
    }
    graph = build_io_graph(cache)
    assert _edges(graph) == {('task:a', 'topic:rs1/mic/audio'), ('topic:rs1/mic/audio', 'task:b')}
    assert _ids(graph, 'topic') == {'topic:rs1/mic/audio'}


def test_wildcard_input_links_matching_output_topics():
    cache = {
        'rs1/vfm': _resp({'a': ('classify', [], ['rs1/vfm/vehicle', 'rs1/vfm/other'])}),
        'edge/gateway': _resp({'g': ('on_vehicle', ['**/vehicle'], [])}),
    }
    edges = _edges(build_io_graph(cache))
    assert ('topic:rs1/vfm/vehicle', 'topic:**/vehicle') in edges
    assert ('topic:**/vehicle', 'task:g') in edges
    assert ('topic:rs1/vfm/other', 'topic:**/vehicle') not in edges


def test_unmatched_input_and_unconsumed_output_stay_visible():
    cache = {'rs1/x': _resp({'a': ('t', ['nobody/publishes'], ['nobody/listens'])})}
    graph = build_io_graph(cache)
    assert _ids(graph, 'topic') == {'topic:nobody/publishes', 'topic:nobody/listens'}
    assert _edges(graph) == {('topic:nobody/publishes', 'task:a'), ('task:a', 'topic:nobody/listens')}


def test_system_tasks_hidden_by_default():
    cache = {'rs1/mic': _resp({'h': ('_heartbeat', [], ['rs1/mic/ctl/heartbeat']), 'a': ('mic_loop', [], ['x'])})}
    assert _ids(build_io_graph(cache), 'task') == {'task:a'}
    assert _ids(build_io_graph(cache, include_system=True), 'task') == {'task:a', 'task:h'}


def test_same_task_name_on_different_apps_stays_separate():
    cache = {
        'rs1/mic': _resp({'id1': ('loop', [], ['x'])}),
        'rs2/mic': _resp({'id2': ('loop', [], ['y'])}),
    }
    graph = build_io_graph(cache)
    assert _ids(graph, 'task') == {'task:id1', 'task:id2'}
    apps = {n['id']: n['app'] for n in graph['nodes'] if n['kind'] == 'task'}
    assert apps == {'task:id1': 'rs1/mic', 'task:id2': 'rs2/mic'}


def test_task_node_carries_snapshot_timestamp():
    cache = {'rs1/mic': _resp({'a': ('loop', [], [])}, timestamp=123)}
    (node,) = build_io_graph(cache)['nodes']
    assert node['timestamp'] == 123


def test_output_is_deterministic():
    cache = {
        'b': _resp({'2': ('t2', ['x'], ['y'])}),
        'a': _resp({'1': ('t1', ['y'], ['x'])}),
    }
    assert build_io_graph(cache) == build_io_graph(dict(reversed(list(cache.items()))))
