"""Vehicle position tracker for AciesOS.

Estimates the position of a single vehicle on a road by observing
when energy peaks pass through each sensor node.

The road is modelled as a polyline projected into 1-D arc-length.
Each sensor node has a known position along the road. As the vehicle
passes nodes, their energy peaks in sequence. A linear regression of
peak times vs node arc positions yields speed and direction:

    t_peak = slope * arc_node + intercept
    speed  = 1 / slope          (m/s, sign gives direction)
    p0     = -intercept / slope (initial position at t=0)

Once the fit has enough observations (``min_peaks``), the tracker
publishes extrapolated positions at 1 Hz. New peaks continuously
refine the fit. On open roads the vehicle bounces at the endpoints;
on loops it wraps around.

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
        # no ambiguity: pick nearest segment
        best = min(candidates, key=lambda c: c[1])
        return best[0]

    # loop: among segments within 2x the best perpendicular distance,
    # pick the one closest to prev_arc along the road
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

    # fallback: last point
    return road[-1][0], road[-1][1]


# --- linear fit ---


def _fit_velocity(peaks: list[tuple[int, Metres]]) -> tuple[float, Metres] | None:
    """Fit a line t = slope * arc + intercept to observed peak times.

    Args:
        peaks: list of (timestamp_ns, node_arc_m) pairs.

    Returns (speed_m_s, position_at_latest_peak_time) or None if fit fails.
    Speed sign encodes direction (positive = increasing arc).
    """
    n = len(peaks)
    if n < 2:
        return None

    # linear regression: t_s = slope * arc + intercept
    # use seconds relative to first peak to avoid precision issues
    t0 = peaks[0][0]
    sum_a = 0.0
    sum_t = 0.0
    sum_at = 0.0
    sum_aa = 0.0
    for ts_ns, arc in peaks:
        t_s = (ts_ns - t0) / _NS_PER_S
        sum_a += arc
        sum_t += t_s
        sum_at += arc * t_s
        sum_aa += arc * arc

    denom = n * sum_aa - sum_a * sum_a
    if abs(denom) < 1e-12:
        return None  # all peaks at same node

    slope = (n * sum_at - sum_a * sum_t) / denom
    intercept = (sum_t - slope * sum_a) / n

    if abs(slope) < 1e-12:
        return None  # near-zero slope -> can't determine speed

    speed = 1.0 / slope  # m/s (sign = direction)
    # position at the latest peak time
    latest_t_s = (peaks[-1][0] - t0) / _NS_PER_S
    pos = speed * (latest_t_s - intercept)

    return speed, pos


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

    # project each node onto the road
    gps_table: dict[str, list[float]] = config.get('gps', {})
    node_arcs: dict[str, float] = {}
    for node, (lat, lon) in gps_table.items():
        node_arcs[node] = _project_onto_road(road, lat, lon)
        logger.info('node %s: lat=%.6f lon=%.6f -> arc=%.1f m', node, lat, lon, node_arcs[node])
    ctx.app['node_arcs'] = node_arcs

    ctx.app['node_mapping'] = config.get('map_node_mapping', {})

    tracker_cfg: dict[str, Any] = config.get('tracker', {})
    ctx.app['min_peaks'] = tracker_cfg.get('min_peaks', 3)
    ctx.app['peak_window_ns'] = int(tracker_cfg.get('peak_window', 30) * _NS_PER_S)

    ctx.app['config_mtime'] = os.path.getmtime(ctx.cfg['config_path'])
    logger.info('tracker min_peaks=%d peak_window=%ds', ctx.app['min_peaks'], tracker_cfg.get('peak_window', 30))


# --- app ---


@app.on_startup
def setup(ctx: AciesContext) -> None:
    _reload_config(ctx)

    ensemble_win = ctx.cfg.get('ensemble_win', 30)
    ctx.app['predictions'] = TimeWindow(window_ns=ensemble_win * _NS_PER_S, data_clock=True)

    # energy tracking: per-node energy for peak detection
    ctx.app['energy'] = TimeWindow(window_ns=5 * _NS_PER_S, data_clock=True)
    ctx.app['prev_peak_node'] = None  # last node that was the loudest
    # peaks: list of (timestamp_ns, arc_m) for the linear fit
    ctx.app['peaks'] = []

    # tracking state (populated once fit succeeds)
    ctx.app['speed'] = 0.0  # m/s, signed
    ctx.app['ref_arc'] = 0.0  # position at ref_time
    ctx.app['ref_time_ns'] = 0  # timestamp of last fit
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

    # --- peak detection: which node is loudest right now? ---
    loudest_node: str | None = None
    loudest_energy = 0.0
    for node in node_arcs:
        entry = energy_win.latest(node)
        if entry is not None:
            _, e = entry
            if e > loudest_energy:
                loudest_energy = e
                loudest_node = node

    if loudest_node is None:
        return

    prev_peak: str | None = ctx.app['prev_peak_node']
    if loudest_node != prev_peak:
        ctx.app['prev_peak_node'] = loudest_node
        if prev_peak is not None:
            # transition detected -> record peak
            peaks: list[tuple[int, Metres]] = ctx.app['peaks']
            peaks.append((ts_ns, node_arcs[loudest_node]))
            logger.info('peak transition: %s -> %s (arc=%.1f m)', prev_peak, loudest_node, node_arcs[loudest_node])


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

    # --- prune old peaks and refit ---
    peaks: list[tuple[int, Metres]] = ctx.app['peaks']
    if peaks:
        peak_window_ns: int = ctx.app['peak_window_ns']
        cutoff = now_ns - peak_window_ns
        peaks = [(t, a) for t, a in peaks if t > cutoff]
        ctx.app['peaks'] = peaks

    min_peaks: int = ctx.app['min_peaks']
    if len(peaks) >= min_peaks:
        result = _fit_velocity(peaks)
        if result is not None:
            speed, pos = result
            ctx.app['speed'] = speed
            ctx.app['ref_arc'] = _wrap_arc(pos, total, is_loop)
            ctx.app['ref_time_ns'] = peaks[-1][0]
            ctx.app['tracking'] = True
            logger.debug('refit: speed=%.1f m/s pos=%.1f m (%d peaks)', speed, ctx.app['ref_arc'], len(peaks))

    if not ctx.app['tracking']:
        return

    # --- extrapolate position ---
    speed = ctx.app['speed']
    ref_arc: Metres = ctx.app['ref_arc']
    ref_time_ns: int = ctx.app['ref_time_ns']

    dt = (now_ns - ref_time_ns) / _NS_PER_S
    arc = ref_arc + speed * dt

    # bounce on open roads
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
