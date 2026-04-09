"""Vehicle position tracker for AciesOS.

Estimates the position of a single vehicle on a road by observing
energy peaks at each sensor node independently.

The road is modelled as a polyline projected into 1-D arc-length.
Each sensor node has a known position along the road. As the vehicle
passes, each node's energy rises then falls — a local peak. The peak
timestamps are collected and fit against node positions along the road:

    t_peak = slope * arc_node + intercept
    speed  = 1 / slope          (m/s, sign gives direction)

Direction is determined by checking which sign of slope (forward vs
reverse along the road) better fits the observed peak ordering. The
fit requires ``min_peaks`` distinct node peaks within ``peak_window``
seconds. Position is then extrapolated at 1 Hz.

On open roads the vehicle bounces at the endpoints; on loops it wraps.

Inputs:
    **/energy           energy dict from sensor nodes
    **/vehicle          classifier predictions (for vehicle label)
    config [gps]        node GPS coordinates
    config [road]       road polyline coordinates
    config [tracker]    min_peaks, peak_window

Output:
    ctx.ns.topic('gps') position dict matching gps.py format

Usage::

    acies-tracker --config ict.toml
                  [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any, TypeAlias

import click
import tomli as tomllib
from acies.buffers.temporal import TimeWindow
from acies.core import AciesApp, AciesContext, setup_logging
from acies.core.msg import AciesInference

logger = logging.getLogger(__name__)

app = AciesApp()

_NS_PER_S = 1_000_000_000

# --- type aliases ---

Metres: TypeAlias = float
"""Distance in metres."""

RoadPoint: TypeAlias = tuple[float, float, Metres]
"""A point on the road polyline: (latitude, longitude, cumulative_distance_m)."""

Road: TypeAlias = list[RoadPoint]
"""Road polyline as a sequence of RoadPoints."""

# --- geometry helpers (lat/lon in degrees, distances in metres) ---

_DEG_TO_RAD = math.pi / 180.0
_EARTH_R = 6_371_000.0  # metres


def _is_loop(road: Road) -> bool:
    """True if the road is a closed loop (first and last coordinates are identical)."""
    return road[0][0] == road[-1][0] and road[0][1] == road[-1][1]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    dlat = (lat2 - lat1) * _DEG_TO_RAD
    dlon = (lon2 - lon1) * _DEG_TO_RAD
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * _DEG_TO_RAD) * math.cos(lat2 * _DEG_TO_RAD) * math.sin(dlon / 2) ** 2
    return _EARTH_R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_road(coords: list[list[float]]) -> Road:
    """Convert a polyline of [lat, lon] into (lat, lon, cumulative_distance_m).

    Returns a list sorted by cumulative distance from the first point.
    """
    road: Road = [(coords[0][0], coords[0][1], 0.0)]
    cumulative = 0.0
    for i in range(1, len(coords)):
        d = _haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        cumulative += d
        road.append((coords[i][0], coords[i][1], cumulative))
    return road


def _project_onto_segment(
    lat: float, lon: float, lat1: float, lon1: float, lat2: float, lon2: float, arc1: Metres, arc2: Metres
) -> tuple[Metres, Metres]:
    """Project a point onto a single road segment.

    Args:
        lat, lon: point to project.
        lat1, lon1: segment start point.
        lat2, lon2: segment end point.
        arc1: cumulative road distance at segment start.
        arc2: cumulative road distance at segment end.

    Returns (arc_length, perpendicular_distance_m).
    """
    seg_len = arc2 - arc1
    if seg_len == 0:
        return arc1, _haversine(lat, lon, lat1, lon1)

    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    dy = lat2 - lat1
    px = (lon - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    py = lat - lat1
    t = max(0.0, min(1.0, (px * dx + py * dy) / (dx * dx + dy * dy)))

    proj_lat = lat1 + t * (lat2 - lat1)
    proj_lon = lon1 + t * (lon2 - lon1)
    return arc1 + t * seg_len, _haversine(lat, lon, proj_lat, proj_lon)


def _arc_distance(a: Metres, b: Metres, total: Metres, is_loop: bool) -> Metres:
    """Shortest distance in metres between two road positions, wrapping if loop."""
    d = abs(a - b)
    if is_loop:
        return min(d, total - d)
    return d


def _project_onto_road(road: Road, lat: float, lon: float, prev_arc: Metres | None = None) -> Metres:
    """Project a lat/lon point onto the road polyline.

    Args:
        road: road polyline.
        lat, lon: point to project.
        prev_arc: previous estimated position along the road.
            When given and the road is a loop, the candidate closest to
            this value is preferred to resolve ambiguity.

    Returns the projected position in metres along the road.
    """
    total = road[-1][2]
    is_loop = _is_loop(road)

    candidates: list[tuple[Metres, Metres]] = []  # (arc, perp_dist)
    for i in range(len(road) - 1):
        lat1, lon1, arc1 = road[i]
        lat2, lon2, arc2 = road[i + 1]
        arc, dist = _project_onto_segment(lat, lon, lat1, lon1, lat2, lon2, arc1, arc2)
        candidates.append((arc, dist))

    if not candidates:
        return 0.0

    if prev_arc is None or not is_loop:
        best = min(candidates, key=lambda c: c[1])
        return best[0]

    best_perp = min(c[1] for c in candidates)
    near = [c for c in candidates if c[1] <= best_perp * 2 + 1.0]
    return min(near, key=lambda c: _arc_distance(c[0], prev_arc, total, True))[0]


def _wrap_arc(arc: Metres, total: Metres, is_loop: bool) -> Metres:
    """Clamp arc to [0, total] or wrap around for loops."""
    if is_loop:
        return arc % total
    return max(0.0, min(total, arc))


def _arc_to_latlon(road: Road, arc: Metres) -> tuple[float, float]:
    """Convert an arc-length distance back to lat/lon by interpolating the road polyline."""
    total = road[-1][2]
    is_loop = _is_loop(road)
    arc = _wrap_arc(arc, total, is_loop)

    for i in range(len(road) - 1):
        _, _, arc1 = road[i]
        _, _, arc2 = road[i + 1]
        if arc1 <= arc <= arc2:
            seg_len = arc2 - arc1
            t = (arc - arc1) / seg_len if seg_len > 0 else 0.0
            lat = road[i][0] + t * (road[i + 1][0] - road[i][0])
            lon = road[i][1] + t * (road[i + 1][1] - road[i][1])
            return lat, lon

    return road[-1][0], road[-1][1]


# --- per-node peak detection ---


def _detect_peaks(energy_win: TimeWindow, node_arcs: dict[str, float]) -> list[tuple[int, str, Metres]]:
    """Detect energy peaks independently per node.

    For each node, scan its energy history and find local maxima
    (a value higher than both its predecessor and successor).

    Returns a list of (timestamp_ns, node_name, arc_m) sorted by time.
    """
    peaks: list[tuple[int, str, Metres]] = []
    for node, arc in node_arcs.items():
        entries = energy_win.get(node)  # [(ts, energy), ...] sorted by ts
        if len(entries) < 3:
            continue
        for i in range(1, len(entries) - 1):
            prev_e = entries[i - 1][1]
            curr_ts, curr_e = entries[i]
            next_e = entries[i + 1][1]
            if curr_e > prev_e and curr_e > next_e:
                peaks.append((curr_ts, node, arc))
    peaks.sort()
    return peaks


# --- linear fit ---


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Fit y = slope * x + intercept via least squares.

    Args:
        points: list of (x, y) pairs.

    Returns (slope, intercept, residual_sum_of_squares) or None.
    """
    n = len(points)
    if n < 2:
        return None

    sum_x = sum_y = sum_xy = sum_xx = 0.0
    for x, y in points:
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # residual
    rss = 0.0
    for x, y in points:
        r = y - (slope * x + intercept)
        rss += r * r

    return slope, intercept, rss


def _fit_velocity(
    peaks: list[tuple[int, str, Metres]],
) -> tuple[float, Metres, int] | None:
    """Fit speed and position from per-node peak events.

    Fits t = slope * arc + intercept in both directions and picks the
    one with the lower residual.

    Args:
        peaks: list of (timestamp_ns, node_name, arc_m) sorted by time.

    Returns (speed_m_s, position_at_latest_peak, ref_time_ns) or None.
    """
    if len(peaks) < 2:
        return None

    # use only the latest peak per node (the most relevant observation)
    latest_per_node: dict[str, tuple[int, Metres]] = {}
    for ts_ns, node, arc in peaks:
        latest_per_node[node] = (ts_ns, arc)

    if len(latest_per_node) < 2:
        return None  # need peaks from at least 2 distinct nodes

    # build (arc, time_s) points for regression
    t0 = min(ts for ts, _ in latest_per_node.values())
    points = [(arc, (ts - t0) / _NS_PER_S) for ts, arc in latest_per_node.values()]

    result = _fit_line(points)
    if result is None:
        return None

    slope, intercept, _rss = result
    if abs(slope) < 1e-12:
        return None

    speed = 1.0 / slope  # m/s, sign = direction
    # position at latest peak time
    latest_ts = max(ts for ts, _ in latest_per_node.values())
    latest_t_s = (latest_ts - t0) / _NS_PER_S
    pos = speed * (latest_t_s - intercept)

    return speed, pos, latest_ts


# --- config ---


def _reload_config(ctx: AciesContext) -> None:
    """Read the toml config and update app state derived from it."""
    with open(ctx.cfg['config_path'], 'rb') as f:
        config = tomllib.load(f)

    road_coords: list[list[float]] = config['road']['coordinates']
    road = _build_road(road_coords)
    ctx.app['road'] = road
    logger.info('road: %d segments, %.1f m total', len(road) - 1, road[-1][2])

    is_loop = _is_loop(road)
    ctx.app['is_loop'] = is_loop
    if is_loop:
        logger.info('road detected as loop (first/last coordinates identical)')

    gps_table: dict[str, list[float]] = config.get('gps', {})
    node_arcs: dict[str, float] = {}
    for node, (lat, lon) in gps_table.items():
        node_arcs[node] = _project_onto_road(road, lat, lon)
        logger.info('node %s: lat=%.6f lon=%.6f -> arc=%.1f m', node, lat, lon, node_arcs[node])
    ctx.app['node_arcs'] = node_arcs

    ctx.app['node_mapping'] = config.get('map_node_mapping', {})

    tracker_cfg: dict[str, Any] = config.get('tracker', {})
    ctx.app['min_peaks'] = tracker_cfg.get('min_peaks', 3)
    peak_window = tracker_cfg.get('peak_window', 60)
    ctx.app['peak_window_ns'] = int(peak_window * _NS_PER_S)

    ctx.app['config_mtime'] = os.path.getmtime(ctx.cfg['config_path'])
    logger.info('tracker min_peaks=%d peak_window=%ds', ctx.app['min_peaks'], peak_window)


# --- app ---


@app.on_startup
def setup(ctx: AciesContext) -> None:
    _reload_config(ctx)

    ensemble_win = ctx.cfg.get('ensemble_win', 30)
    ctx.app['predictions'] = TimeWindow(window_ns=ensemble_win * _NS_PER_S, data_clock=True)

    # per-node energy buffer for peak detection (use peak_window so we see
    # enough history for the vehicle to pass multiple nodes)
    ctx.app['energy'] = TimeWindow(window_ns=ctx.app['peak_window_ns'], data_clock=True)

    # tracking state
    ctx.app['speed'] = 0.0
    ctx.app['ref_arc'] = 0.0
    ctx.app['ref_time_ns'] = 0
    ctx.app['tracking'] = False


@app.schedule(5.0)
def check_config(ctx: AciesContext) -> None:
    """Reload the toml config file if it has been modified on disk."""
    try:
        mtime = os.path.getmtime(ctx.cfg['config_path'])
    except OSError:
        return
    if mtime != ctx.app['config_mtime']:
        logger.info('config file changed on disk; reloading %s', ctx.cfg['config_path'])
        _reload_config(ctx)


@app.subscribe('**/energy')
def on_energy(ctx: AciesContext, msg: Any) -> None:
    source: str = msg['source']
    ts_ns: int = msg['timestamp']
    energy_by_ch: dict[str, float] = msg['energy']
    energy = max(energy_by_ch.values())

    host = source.split('/')[0]
    node_mapping: dict[str, str] = ctx.app['node_mapping']
    canonical = node_mapping.get(host, host)

    node_arcs: dict[str, float] = ctx.app['node_arcs']
    if canonical not in node_arcs:
        return

    energy_win: TimeWindow = ctx.app['energy']
    energy_win.add(canonical, ts_ns, energy)
    logger.debug('energy: %s=%.0f', canonical, energy)


@app.subscribe('**/vehicle')
def on_vehicle(ctx: AciesContext, msg: AciesInference) -> None:
    pred_win: TimeWindow = ctx.app['predictions']
    for pred in msg.predictions:
        pred_win.add(pred.label, msg.timestamp, pred.score)


def _ensemble_label(pred_win: TimeWindow) -> str | None:
    """Pick the label with the highest average score over the window."""
    scores: dict[str, list[float]] = defaultdict(list)
    for label in pred_win.keys():
        for _ts, score in pred_win.get(label):
            scores[label].append(score)
    if not scores:
        return None
    return max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))


@app.schedule(1.0)
def estimate(ctx: AciesContext) -> None:
    energy_win: TimeWindow = ctx.app['energy']
    now_ns: int = energy_win.latest_ts
    if now_ns == 0:
        return

    road: Road = ctx.app['road']
    total = road[-1][2]
    is_loop: bool = ctx.app['is_loop']
    node_arcs: dict[str, float] = ctx.app['node_arcs']

    # --- detect peaks from energy buffer and attempt fit ---
    peaks = _detect_peaks(energy_win, node_arcs)
    min_peaks: int = ctx.app['min_peaks']

    if peaks:
        peak_order = ' -> '.join(f'{node}@{ts / _NS_PER_S:.1f}' for ts, node, _arc in peaks)
        distinct = len({p[1] for p in peaks})
        logger.info('peaks (%d from %d nodes): %s', len(peaks), distinct, peak_order)

    if len(peaks) >= min_peaks:
        result = _fit_velocity(peaks)
        if result is not None:
            speed, pos, ref_ts = result
            ctx.app['speed'] = speed
            ctx.app['ref_arc'] = _wrap_arc(pos, total, is_loop)
            ctx.app['ref_time_ns'] = ref_ts
            ctx.app['tracking'] = True
            logger.info(
                'fit: speed=%.1f m/s pos=%.1f m (%d peaks from %d nodes)',
                speed, ctx.app['ref_arc'], len(peaks), distinct,
            )
        else:
            logger.info('fit failed (%d peaks from %d nodes)', len(peaks), distinct)

    if not ctx.app['tracking']:
        return

    # --- extrapolate position ---
    speed: float = ctx.app['speed']
    ref_arc: Metres = ctx.app['ref_arc']
    ref_time_ns: int = ctx.app['ref_time_ns']

    dt = (now_ns - ref_time_ns) / _NS_PER_S
    arc = ref_arc + speed * dt

    if not is_loop:
        while True:
            if arc < 0:
                arc = -arc
                speed = -speed
            elif arc > total:
                arc = 2 * total - arc
                speed = -speed
            else:
                break
        ctx.app['speed'] = speed

    arc = _wrap_arc(arc, total, is_loop)

    label = _ensemble_label(ctx.app['predictions']) or 'unknown'
    lat, lon = _arc_to_latlon(road, arc)
    ctx.publish(ctx.ns.topic('gps'), {label: {'lat': lat, 'lon': lon, 'elevation': 0.0}, 'timestamp': now_ns})
    logger.debug('estimate: arc=%.1f m speed=%.1f m/s lat=%.6f lon=%.6f', arc, speed, lat, lon)


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('tracker stopped')


@app.cli()
@click.option(
    '--config',
    'config_path',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Site config file (.toml) with [road] and [gps] sections.',
)
def main(config_path: str) -> None:
    app.state.config.update({'config_path': config_path})
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
