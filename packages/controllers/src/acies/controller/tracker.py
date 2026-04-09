"""Vehicle position tracker for AciesOS.

Estimates the position of a single vehicle on a road segment by fusing
energy readings from multiple sensor nodes. Each node's energy (std dev
of geophone samples) is used as a proximity signal; higher energy means
the vehicle is closer to that node.

The road is modelled as a polyline. Node positions and the polyline are
projected into a 1-D coordinate (arc-length along the road). An energy-
weighted average of node positions gives the estimated vehicle location,
which is then converted back to lat/lon and published in the same format
as gps.py.

Between energy updates the position is extrapolated using the last
estimated speed and direction.

Inputs:
    **/energy           energy dict from sensor nodes
    **/vehicle          classifier predictions (for vehicle label)
    config [gps]        node GPS coordinates
    config [road]       road polyline coordinates

Output:
    ctx.ns.topic('gps') position dict matching gps.py format

Usage::

    acies-tracker --config ict.toml
                  [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import math
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


def _shortest_offset(a: Metres, b: Metres, total: Metres, is_loop: bool) -> Metres:
    """Signed offset from a to b, taking the shorter path on loops."""
    offset = b - a
    if is_loop:
        if offset > total / 2:
            offset -= total
        elif offset < -total / 2:
            offset += total
    return offset


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


@app.on_startup
def setup(ctx: AciesContext) -> None:
    with open(ctx.cfg['config_path'], 'rb') as f:
        config = tomllib.load(f)

    road_coords: list[list[float]] = config['road']['coordinates']
    road = _build_road(road_coords)
    ctx.app['road'] = road
    logger.info('road: %d segments, %.1f m total', len(road) - 1, road[-1][2])

    is_loop = _is_loop(road)
    ctx.app['is_loop'] = is_loop
    if is_loop:
        logger.info('road detected as loop (first/last point <1m apart)')

    # project each node onto the road
    gps_table: dict[str, list[float]] = config.get('gps', {})
    node_arcs: dict[str, float] = {}
    for node, (lat, lon) in gps_table.items():
        node_arcs[node] = _project_onto_road(road, lat, lon)
        logger.info('node %s: lat=%.6f lon=%.6f -> arc=%.1f m', node, lat, lon, node_arcs[node])
    ctx.app['node_arcs'] = node_arcs

    # map_node_mapping: maps alternative node ids (e.g. gq-1) to canonical ids (e.g. rs1)
    ctx.app['node_mapping'] = config.get('map_node_mapping', {})

    # tracker state
    ctx.app['energy'] = TimeWindow(window_ns=5 * _NS_PER_S, data_clock=True)
    ensemble_win = ctx.cfg.get('ensemble_win', 30)
    ctx.app['predictions'] = TimeWindow(window_ns=ensemble_win * _NS_PER_S, data_clock=True)
    ctx.app['est_arc'] = road[-1][2] / 2  # start at midpoint
    ctx.app['est_speed'] = 0.0  # m/s along road
    ctx.app['last_update_ns'] = 0


@app.subscribe('**/energy')
def on_energy(ctx: AciesContext, msg: Any) -> None:
    source: str = msg['source']
    ts_ns: int = msg['timestamp']
    energy_by_ch: dict[str, float] = msg['energy']
    energy = max(energy_by_ch.values())

    # extract host name from source (e.g. "rs1/geo" -> "rs1", "gq-2/geo" -> "gq-2")
    host = source.split('/')[0]

    # map alternative node id to canonical id if needed
    node_mapping: dict[str, str] = ctx.app['node_mapping']
    canonical = node_mapping.get(host, host)

    node_arcs: dict[str, float] = ctx.app['node_arcs']
    if canonical not in node_arcs:
        return

    energy_win: TimeWindow = ctx.app['energy']
    energy_win.add(canonical, ts_ns, energy)


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
    node_arcs: dict[str, float] = ctx.app['node_arcs']
    road: Road = ctx.app['road']

    energy_win: TimeWindow = ctx.app['energy']
    now_ns: int = energy_win.latest_ts  # use data clock as time source
    if now_ns == 0:
        return  # no data received yet

    fresh: dict[str, tuple[int, float]] = {}
    for node in node_arcs:
        entry = energy_win.latest(node)
        if entry is not None:
            fresh[node] = entry

    total = road[-1][2]
    is_loop: bool = ctx.app['is_loop']
    last_ns: int = ctx.app['last_update_ns']

    label = _ensemble_label(ctx.app['predictions']) or 'unknown'

    if not fresh:
        # no recent data -> extrapolate from last known state
        if last_ns > 0:
            dt = (now_ns - last_ns) / _NS_PER_S
            ctx.app['est_arc'] = _wrap_arc(ctx.app['est_arc'] + ctx.app['est_speed'] * dt, total, is_loop)
            ctx.app['last_update_ns'] = now_ns
        lat, lon = _arc_to_latlon(road, ctx.app['est_arc'])
        ctx.publish(ctx.ns.topic('gps'), {label: {'lat': lat, 'lon': lon, 'elevation': 0.0}, 'timestamp': now_ns})
        return

    # energy-weighted position estimate
    # on a loop, compute offsets relative to current estimate to avoid
    # averaging across the wraparound boundary
    prev_arc = ctx.app['est_arc']
    total_weight = 0.0
    weighted_offset = 0.0
    for node, (_ts, e) in fresh.items():
        w = e * e  # square to sharpen the peak
        weighted_offset += w * _shortest_offset(prev_arc, node_arcs[node], total, is_loop)
        total_weight += w
    new_arc = prev_arc + weighted_offset / total_weight if total_weight > 0 else prev_arc
    new_arc = _wrap_arc(new_arc, total, is_loop)

    # estimate speed from position change
    if last_ns > 0:
        dt = (now_ns - last_ns) / _NS_PER_S
        if dt > 0:
            ctx.app['est_speed'] = _shortest_offset(prev_arc, new_arc, total, is_loop) / dt

    ctx.app['est_arc'] = new_arc
    ctx.app['last_update_ns'] = now_ns

    lat, lon = _arc_to_latlon(road, new_arc)
    ctx.publish(ctx.ns.topic('gps'), {label: {'lat': lat, 'lon': lon, 'elevation': 0.0}, 'timestamp': now_ns})
    logger.debug('estimate: arc=%.1f m speed=%.1f m/s lat=%.6f lon=%.6f', new_arc, ctx.app['est_speed'], lat, lon)


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
