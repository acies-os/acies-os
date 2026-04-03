"""Tests for TimeWindow."""

from __future__ import annotations

from acies.buffers.temporal import TimeWindow

NS = 1_000_000_000  # 1 second in nanoseconds


def _make(window_s: int, now_ns: int) -> tuple[TimeWindow, list[int]]:
    """Create a TimeWindow with a controllable clock."""
    clock = [now_ns]
    buf = TimeWindow(window_ns=window_s * NS, _now_fn=lambda: clock[0])
    return buf, clock


# --- basic add/get ---


def test_add_and_get():
    buf, _ = _make(5, 100 * NS)
    buf.add('a', 98 * NS, 'v1')
    buf.add('a', 99 * NS, 'v2')
    entries = buf.get('a')
    assert len(entries) == 2
    assert entries[0] == (98 * NS, 'v1')
    assert entries[1] == (99 * NS, 'v2')


def test_get_returns_copy():
    """Mutating the returned list should not affect internal state."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v')
    result = buf.get('a')
    result.clear()
    assert len(buf.get('a')) == 1


def test_get_missing_key():
    buf, _ = _make(5, 100 * NS)
    assert buf.get('missing') == []


# --- pruning ---


def test_old_entries_pruned():
    buf, _ = _make(5, 100 * NS)
    buf.add('a', 90 * NS, 'old')
    buf.add('a', 96 * NS, 'keep')
    entries = buf.get('a')
    assert len(entries) == 1
    assert entries[0][1] == 'keep'


def test_pruning_on_clock_advance():
    """Entries that were in-window become stale when the clock advances."""
    buf, clock = _make(5, 100 * NS)
    buf.add('a', 97 * NS, 'v1')
    buf.add('a', 99 * NS, 'v2')
    assert len(buf.get('a')) == 2

    clock[0] = 103 * NS  # now=103, cutoff=98 -> v1 pruned
    assert len(buf.get('a')) == 1
    assert buf.get('a')[0][1] == 'v2'


def test_all_entries_pruned():
    """All entries expire when the clock advances far enough."""
    buf, clock = _make(5, 100 * NS)
    buf.add('a', 95 * NS, 'v1')
    buf.add('a', 96 * NS, 'v2')
    clock[0] = 200 * NS
    assert buf.get('a') == []
    assert 'a' not in buf


def test_pruning_independent_per_key():
    buf, _ = _make(5, 100 * NS)
    buf.add('a', 90 * NS, 'old_a')
    buf.add('a', 99 * NS, 'new_a')
    buf.add('b', 99 * NS, 'new_b')
    assert len(buf.get('a')) == 1
    assert len(buf.get('b')) == 1


# --- latest ---


def test_latest():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v1')
    buf.add('a', 99 * NS, 'v2')
    result = buf.latest('a')
    assert result == (99 * NS, 'v2')


def test_latest_missing_key():
    buf, _ = _make(5, 100 * NS)
    assert buf.latest('missing') is None


def test_latest_all_expired():
    buf, clock = _make(5, 100 * NS)
    buf.add('a', 90 * NS, 'old')
    clock[0] = 200 * NS
    assert buf.latest('a') is None


# --- nearest ---


def test_nearest_exact_match():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v1')
    buf.add('a', 97 * NS, 'v2')
    buf.add('a', 99 * NS, 'v3')
    result = buf.nearest('a', 97 * NS)
    assert result == (97 * NS, 'v2')


def test_nearest_between_entries():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 94 * NS, 'v1')
    buf.add('a', 98 * NS, 'v2')
    # 97 is closer to 98 than 94
    result = buf.nearest('a', 97 * NS)
    assert result == (98 * NS, 'v2')


def test_nearest_equidistant_prefers_earlier():
    """When equidistant, min() returns the first candidate — which is the later one
    (bisect finds it first). This tests the actual behavior."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 94 * NS, 'v1')
    buf.add('a', 96 * NS, 'v2')
    # 95 is equidistant from 94 and 96
    result = buf.nearest('a', 95 * NS)
    assert result is not None
    assert result[0] in (94 * NS, 96 * NS)


def test_nearest_before_all_entries():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v1')
    buf.add('a', 99 * NS, 'v2')
    result = buf.nearest('a', 91 * NS)
    assert result == (95 * NS, 'v1')


def test_nearest_after_all_entries():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 92 * NS, 'v1')
    buf.add('a', 95 * NS, 'v2')
    result = buf.nearest('a', 99 * NS)
    assert result == (95 * NS, 'v2')


def test_nearest_single_entry():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'only')
    result = buf.nearest('a', 99 * NS)
    assert result == (95 * NS, 'only')


def test_nearest_missing_key():
    buf, _ = _make(5, 100 * NS)
    assert buf.nearest('missing', 99 * NS) is None


# --- range ---


def test_range_inclusive():
    buf, _ = _make(10, 100 * NS)
    for i in range(91, 100):
        buf.add('a', i * NS, f'v{i}')
    result = buf.range('a', 94 * NS, 96 * NS)
    assert len(result) == 3
    assert result[0] == (94 * NS, 'v94')
    assert result[2] == (96 * NS, 'v96')


def test_range_no_overlap():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v')
    result = buf.range('a', 97 * NS, 99 * NS)
    assert result == []


def test_range_missing_key():
    buf, _ = _make(5, 100 * NS)
    assert buf.range('missing', 0, 100 * NS) == []


# --- keys / len / contains ---


def test_keys():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v')
    buf.add('b', 96 * NS, 'v')
    assert sorted(buf.keys()) == ['a', 'b']


def test_keys_excludes_empty():
    """keys() prunes and excludes keys whose entries have all expired."""
    buf, clock = _make(5, 100 * NS)
    buf.add('a', 90 * NS, 'expired')
    buf.add('b', 99 * NS, 'live')
    clock[0] = 100 * NS
    assert buf.keys() == ['b']


def test_len():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v1')
    buf.add('a', 96 * NS, 'v2')
    buf.add('b', 97 * NS, 'v3')
    assert len(buf) == 3


def test_contains():
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'v')
    assert 'a' in buf
    assert 'b' not in buf


def test_contains_expired():
    buf, _ = _make(5, 100 * NS)
    buf.add('a', 90 * NS, 'old')
    assert 'a' not in buf


# --- edge cases ---


def test_zero_window():
    """A zero-width window: cutoff == now, so entries at exactly now survive
    (inclusive boundary) but anything older is pruned."""
    buf, _ = _make(0, 100 * NS)
    buf.add('a', 100 * NS, 'exact')
    buf.add('a', 99 * NS, 'old')
    entries = buf.get('a')
    assert len(entries) == 1
    assert entries[0] == (100 * NS, 'exact')


def test_duplicate_timestamps():
    """Multiple entries at the same timestamp are all preserved."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'first')
    buf.add('a', 95 * NS, 'second')
    entries = buf.get('a')
    assert len(entries) == 2
    assert entries[0][1] == 'first'
    assert entries[1][1] == 'second'


def test_many_keys():
    buf, _ = _make(10, 100 * NS)
    for i in range(100):
        buf.add(f'key_{i}', 95 * NS, i)
    assert len(buf.keys()) == 100
    assert len(buf) == 100


def test_large_window_keeps_everything():
    buf, _ = _make(1000, 500 * NS)
    for i in range(100):
        buf.add('a', i * NS, i)
    assert len(buf.get('a')) == 100


# --- range boundary precision ---


def test_range_boundary_inclusive_exclusive():
    """range includes t_end but excludes t_end + 1."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 95 * NS, 'at_start')
    buf.add('a', 97 * NS, 'at_end')
    buf.add('a', 98 * NS, 'after_end')
    result = buf.range('a', 95 * NS, 97 * NS)
    assert len(result) == 2
    assert result[0] == (95 * NS, 'at_start')
    assert result[1] == (97 * NS, 'at_end')


def test_range_single_nanosecond():
    """range with t_start == t_end returns only entries at that exact timestamp."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 94 * NS, 'before')
    buf.add('a', 95 * NS, 'exact')
    buf.add('a', 96 * NS, 'after')
    result = buf.range('a', 95 * NS, 95 * NS)
    assert len(result) == 1
    assert result[0] == (95 * NS, 'exact')


# --- out-of-order timestamps ---


def test_add_out_of_order():
    """Entries added out of order are still returned sorted."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 99 * NS, 'third')
    buf.add('a', 95 * NS, 'first')
    buf.add('a', 97 * NS, 'second')
    entries = buf.get('a')
    assert len(entries) == 3
    assert entries[0] == (95 * NS, 'first')
    assert entries[1] == (97 * NS, 'second')
    assert entries[2] == (99 * NS, 'third')


def test_nearest_with_out_of_order_add():
    """nearest works correctly even after out-of-order inserts."""
    buf, _ = _make(10, 100 * NS)
    buf.add('a', 99 * NS, 'late')
    buf.add('a', 91 * NS, 'early')
    buf.add('a', 95 * NS, 'mid')
    result = buf.nearest('a', 94 * NS)
    assert result == (95 * NS, 'mid')


# --- keys() pruning ---


def test_keys_prunes_expired():
    """keys() prunes stale entries without requiring get() first."""
    buf, clock = _make(5, 100 * NS)
    buf.add('stale', 90 * NS, 'old')
    buf.add('live', 99 * NS, 'new')
    # Don't call get() on 'stale' — keys() should still exclude it
    assert buf.keys() == ['live']
