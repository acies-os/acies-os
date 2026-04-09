"""Tests for namespace.py — topic construction, validation, and wildcard matching."""

import pytest

from acies.core.namespace import Namespace, matches

# ---------------------------------- fixtures ----------------------------------


@pytest.fixture
def ns() -> Namespace:
    return Namespace('edge-01', 'mic')


# --------------------------- namespace construction ---------------------------


def test_namespace_attributes(ns: Namespace) -> None:
    assert ns.namespace == 'edge-01'
    assert ns.name == 'mic'


def test_namespace_ctl_pre_computed(ns: Namespace) -> None:
    assert ns.ctl.kv == 'edge-01/mic/ctl/kv'
    assert ns.ctl.heartbeat == 'edge-01/mic/ctl/heartbeat'
    assert ns.ctl.route == 'edge-01/mic/ctl/route'


# ----------------------------------- topic -----------------------------------


def test_topic_default_prefix(ns: Namespace) -> None:
    assert ns.topic('audio') == 'edge-01/mic/audio'


def test_topic_default_prefix_multiple_parts(ns: Namespace) -> None:
    assert ns.topic('audio', 'raw') == 'edge-01/mic/audio/raw'


def test_topic_default_prefix_no_parts(ns: Namespace) -> None:
    assert ns.topic() == 'edge-01/mic'


def test_topic_no_prefix_empty_string(ns: Namespace) -> None:
    assert ns.topic('building', 'a', 'temperature', prefix='') == 'building/a/temperature'


def test_topic_no_prefix_false(ns: Namespace) -> None:
    assert ns.topic('audio', prefix=False) == 'audio'


def test_topic_prefix_true_is_default(ns: Namespace) -> None:
    assert ns.topic('audio') == ns.topic('audio', prefix=True)


def test_topic_prefix_str(ns: Namespace) -> None:
    assert ns.topic('audio', prefix='org/site-a') == 'org/site-a/audio'


def test_topic_single_wildcard(ns: Namespace) -> None:
    assert ns.topic('*') == 'edge-01/mic/*'


def test_topic_double_wildcard(ns: Namespace) -> None:
    assert ns.topic('**') == 'edge-01/mic/**'


def test_topic_wildcard_no_prefix(ns: Namespace) -> None:
    assert ns.topic('*', prefix='') == '*'


def test_topic_dollar_star(ns: Namespace) -> None:
    assert ns.topic('thermo$*') == 'edge-01/mic/thermo$*'


def test_topic_dollar_star_mixed(ns: Namespace) -> None:
    assert ns.topic('$*sensor') == 'edge-01/mic/$*sensor'


# ----------------------------------- ctl() -----------------------------------


def test_ctl_callable_single_part(ns: Namespace) -> None:
    assert ns.ctl('status') == 'edge-01/mic/ctl/status'


def test_ctl_callable_multiple_parts(ns: Namespace) -> None:
    assert ns.ctl('my', 'service') == 'edge-01/mic/ctl/my/service'


def test_ctl_single_wildcard(ns: Namespace) -> None:
    assert ns.ctl('*') == 'edge-01/mic/ctl/*'


def test_ctl_double_wildcard(ns: Namespace) -> None:
    assert ns.ctl('**') == 'edge-01/mic/ctl/**'


def test_ctl_dollar_star(ns: Namespace) -> None:
    assert ns.ctl('svc$*') == 'edge-01/mic/ctl/svc$*'


# ----------------------- validation: namespace and name -----------------------


def test_empty_namespace_raises() -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        _ = Namespace('', 'mic')


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        _ = Namespace('edge-01', '')


def test_hierarchical_namespace_allowed() -> None:
    ns = Namespace('edge-01/sensor', 'mic')
    assert ns.namespace == 'edge-01/sensor'
    assert ns.name == 'mic'
    assert ns.base == 'edge-01/sensor/mic'
    assert ns.ctl.heartbeat == 'edge-01/sensor/mic/ctl/heartbeat'
    assert ns.topic('audio') == 'edge-01/sensor/mic/audio'


def test_deep_hierarchical_namespace_allowed() -> None:
    ns = Namespace('org/site-a/edge-01/sensor', 'mic')
    assert ns.base == 'org/site-a/edge-01/sensor/mic'


def test_namespace_empty_segment_raises() -> None:
    with pytest.raises(ValueError, match='empty segment'):
        _ = Namespace('edge-01//sensor', 'mic')


def test_slash_in_name_raises() -> None:
    with pytest.raises(ValueError, match="must not contain '/'"):
        _ = Namespace('edge-01', 'my/mic')


@pytest.mark.parametrize('char', ['*', '$', '?', '#'])
def test_forbidden_char_in_namespace_raises(char: str) -> None:
    with pytest.raises(ValueError, match='forbidden characters'):
        _ = Namespace(f'edge{char}01', 'mic')


@pytest.mark.parametrize('char', ['*', '$', '?', '#'])
def test_forbidden_char_in_name_raises(char: str) -> None:
    with pytest.raises(ValueError, match='forbidden characters'):
        _ = Namespace('edge-01', f'mi{char}c')


# -------------------------- validation: topic parts --------------------------


def test_topic_empty_part_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        _ = ns.topic('audio', '', 'raw')


def test_topic_slash_in_part_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError, match="must not contain '/'"):
        _ = ns.topic('audio/raw')


@pytest.mark.parametrize('char', ['?', '#'])
def test_topic_selector_char_raises(ns: Namespace, char: str) -> None:
    with pytest.raises(ValueError, match='forbidden characters'):
        _ = ns.topic(f'audio{char}')


def test_topic_bare_star_in_segment_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError):
        _ = ns.topic('audio*')


def test_topic_bare_dollar_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError):
        _ = ns.topic('audio$')


# --------------------------- validation: ctl parts ---------------------------


def test_ctl_empty_part_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        _ = ns.ctl('')


def test_ctl_slash_in_part_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError, match="must not contain '/'"):
        _ = ns.ctl('my/service')


@pytest.mark.parametrize('char', ['?', '#'])
def test_ctl_selector_char_raises(ns: Namespace, char: str) -> None:
    with pytest.raises(ValueError, match='forbidden characters'):
        _ = ns.ctl(f'serv{char}ce')


def test_ctl_bare_star_in_segment_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError):
        _ = ns.ctl('svc*name')


def test_ctl_bare_dollar_raises(ns: Namespace) -> None:
    with pytest.raises(ValueError):
        _ = ns.ctl('svc$name')


# ----------------------- topic() cross-namespace patterns --------------------


def test_topic_cross_namespace_wildcard(ns: Namespace) -> None:
    assert ns.topic('edge-01', '*', 'audio', prefix='') == 'edge-01/*/audio'


def test_topic_cross_namespace_double_wildcard(ns: Namespace) -> None:
    assert ns.topic('**', 'ctl', 'heartbeat', prefix='') == '**/ctl/heartbeat'


def test_topic_cross_namespace_dollar_star(ns: Namespace) -> None:
    assert ns.topic('edge-01', 'thermo$*', 'temp', prefix='') == 'edge-01/thermo$*/temp'


# ------------------------------- matches: exact -------------------------------


def test_matches_exact() -> None:
    assert matches('edge-01/mic/audio', 'edge-01/mic/audio')


def test_matches_exact_no_match() -> None:
    assert not matches('edge-01/mic/audio', 'edge-01/mic/video')


# --------------------------------- matches: * ---------------------------------


def test_matches_single_wildcard_one_segment() -> None:
    assert matches('edge-01/*/audio', 'edge-01/mic/audio')


def test_matches_single_wildcard_no_match_multiple_segments() -> None:
    assert not matches('edge-01/*/audio', 'edge-01/mic/raw/audio')


def test_matches_single_wildcard_no_match_empty_segment() -> None:
    assert not matches('edge-01/*/audio', 'edge-01//audio')


# -------------------------------- matches: ** --------------------------------


def test_matches_double_wildcard_zero_segments() -> None:
    assert matches('edge-01/mic/**', 'edge-01/mic/')


def test_matches_double_wildcard_one_segment() -> None:
    assert matches('edge-01/mic/**', 'edge-01/mic/audio')


def test_matches_double_wildcard_multiple_segments() -> None:
    assert matches('edge-01/mic/**', 'edge-01/mic/audio/raw')


def test_matches_double_wildcard_prefix_no_match() -> None:
    assert not matches('edge-01/mic/**', 'edge-02/mic/audio')
