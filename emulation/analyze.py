"""Summarize an emulation record: localization error and data delivery per fault phase.

Reads a recorder JSONL file (default: the newest in ~/.acies/emu/records) and
looks at one deploy (default: the last one). Phases are the intervals between
the experiment's event times, measured from the replay start.

For each phase it reports:

* SPAR windows: how many 2-second windows produced a localization (out of the
  number the phase spans), the mean number of nodes that fed them, and the
  localization error against GPS truth at the window start (median / mean / p90).
* Delivery: per node, the share of 1-second geo + mic windows that reached the edge.

Also writes ``<record>.windows.csv`` with one row per SPAR output.

Usage::

    python /emu/analyze.py [RECORD.jsonl] [--deploy N]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

_NS = 1_000_000_000
_WINDOW_S = 2  # SPAR window length
_EARTH_R = 6_371_000.0


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Equirectangular distance; accurate to well under 1 m at this scale."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return _EARTH_R * math.hypot(x, y)


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def load(path: Path, deploy: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    armed = [r for r in records if r['type'] == 'event' and r.get('action') == 'armed']
    if not armed:
        raise SystemExit(f'{path}: no deploy found (no "armed" event); was topo.py connected to the edge?')
    arm = armed[deploy]
    start_ns = int(arm['start_at'] * _NS)
    later = [r for r in armed if r['wall_ns'] > arm['wall_ns']]
    end_ns = later[0]['wall_ns'] if later else math.inf
    window = [r for r in records if start_ns <= r['wall_ns'] < end_ns]
    return arm, window


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('record', nargs='?', help='recorder JSONL file (default: newest)')
    parser.add_argument('--deploy', type=int, default=-1, help='which deploy in the file (0 = first, -1 = last)')
    args = parser.parse_args()

    if args.record:
        path = Path(args.record)
    else:
        files = sorted(Path('~/.acies/emu/records').expanduser().glob('*.jsonl'))
        if not files:
            raise SystemExit('no record files in ~/.acies/emu/records')
        path = files[-1]

    arm, records = load(path, args.deploy)
    events: list[dict[str, Any]] = arm.get('events', [])

    # --- clocks ---
    # Replays share a data origin (acies-replay --origin run): the run's earliest timestamp plays
    # at start_at. data0 is the earliest data time that reached the edge, normally that origin,
    # so t = data_s - data0 is "seconds into the replay" for SPAR windows. Delivery is measured
    # on the wall clock (when did the data cross the link).
    start_ns = int(arm['start_at'] * _NS)
    energy = [r for r in records if r['type'] == 'energy' and r.get('data_ns')]
    if not energy:
        raise SystemExit('no node data reached the edge during this deploy')
    data0_s = min(r['data_ns'] for r in energy) // _NS
    duration_s = int((max(r['wall_ns'] for r in energy) - start_ns) // _NS) + 1

    truth: dict[int, tuple[float, float]] = {}
    for r in records:
        if r['type'] == 'truth' and r.get('data_ns'):
            truth[int(r['data_ns'] // _NS)] = (r['lat'], r['lon'])

    def truth_at(sec: int) -> tuple[float, float] | None:
        for d in (0, 1, -1):
            if sec + d in truth:
                return truth[sec + d]
        return None

    # --- phases between event times ---
    cuts = sorted({0.0, *(float(e['at']) for e in events if float(e['at']) < duration_s), float(duration_s)})
    phases = list(itertools.pairwise(cuts))

    def phase_of(t: float) -> int | None:
        for i, (a, b) in enumerate(phases):
            if a <= t < b:
                return i
        return None

    def label(i: int) -> str:
        a, b = phases[i]
        acts = [f'{e["action"]} {e.get("nodes", e.get("node"))}' for e in events if float(e['at']) == a]
        return f'{a:>5.0f}-{b:<5.0f} ' + ('; '.join(acts) if acts else 'baseline')

    # --- SPAR windows ---
    rows: list[dict[str, Any]] = []
    for r in records:
        if r['type'] != 'spar' or r.get('window_start_s') is None:
            continue
        ws = int(r['window_start_s'])
        t = ws - data0_s
        gt = truth_at(ws)
        err = _dist_m(r['lat'], r['lon'], *gt) if gt and r.get('lat') is not None else None
        # output time minus the wall time the window's data finished playing
        lag = (r['wall_ns'] - start_ns) / _NS - (t + _WINDOW_S)
        rows.append(
            {
                't_s': t,
                'phase': phase_of(t),
                'lag_s': round(lag, 2),
                'window_start_s': ws,
                'label': r['label'],
                'score': round(r['score'], 4),
                'n_nodes': len(r['nodes']),
                'nodes': ' '.join(r['nodes']),
                'lat': r['lat'],
                'lon': r['lon'],
                'truth_lat': gt[0] if gt else None,
                'truth_lon': gt[1] if gt else None,
                'error_m': round(err, 2) if err is not None else None,
            }
        )

    # --- per-node delivery: (node, phase) -> 1-s windows that arrived while the phase was active ---
    nodes = sorted({r['node'] for r in energy}, key=lambda n: (len(n), n))
    got: dict[tuple[str, int], int] = defaultdict(int)
    for r in energy:
        p = phase_of((r['wall_ns'] - start_ns) / _NS)
        if p is not None:
            got[(r['node'], p)] += 1

    print(f'record:  {path}')
    print(f'deploy:  start_at={arm["start_at"]:.3f}  played={duration_s}s  events={len(events)}')
    print()
    print(f'{"phase (s)":<40} {"windows":>9} {"nodes":>6} {"lag s":>6} {"err med":>8} {"mean":>7} {"p90":>7}')
    for i in range(len(phases)):
        a, b = phases[i]
        ph = [x for x in rows if x['phase'] == i]
        errs = [x['error_m'] for x in ph if x['error_m'] is not None]
        expected = max(1, int((b - a) // _WINDOW_S))
        mean_nodes = statistics.mean(x['n_nodes'] for x in ph) if ph else 0.0
        lag = f'{statistics.mean(x["lag_s"] for x in ph):>6.1f}' if ph else '     -'
        err_s = (
            f'{statistics.median(errs):>8.1f} {statistics.mean(errs):>7.1f} {_pct(errs, 0.9):>7.1f}'
            if errs
            else '       -       -       -'
        )
        print(f'{label(i):<40} {len(ph):>4}/{expected:<4} {mean_nodes:>6.1f} {lag} {err_s}')
    print()
    print('share of expected geo+mic windows that reached the edge during each phase (wall clock):')
    print(f'{"phase (s)":<14}' + ''.join(f'{n:>6}' for n in nodes))
    for i, (a, b) in enumerate(phases):
        expected = 2 * max(1, int(b - a))
        cells = ''.join(f'{100 * got[(n, i)] / expected:>5.0f}%' for n in nodes)
        print(f'{a:>5.0f}-{b:<7.0f} {cells}')

    out = path.with_suffix('.windows.csv')
    if rows:
        with out.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nper-window rows: {out}')


if __name__ == '__main__':
    main()
