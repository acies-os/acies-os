"""Vehicle position tracker for AciesOS.

Simplified tracker: starts at the beginning of the road, moves at a
constant configured speed in a configured direction. Subscribes to
energy and vehicle topics for future refinement but currently does
not use them for position estimation.

The road is modelled as a polyline projected into 1-D arc-length.
Position is extrapolated at 1 Hz. On open roads the vehicle bounces
at the endpoints; on loops it wraps around.

Inputs:
    **/energy           energy dict from sensor nodes (logged for tuning)
    **/vehicle          classifier predictions (for vehicle label)
    config [road]       road polyline coordinates
    config [tracker]    speed, direction

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

# --- geometry helpers ---

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
    """Convert a polyline of [lat, lon] into (lat, lon, cumulative_distance_m)."""
    road: Road = [(coords[0][0], coords[0][1], 0.0)]
    cumulative = 0.0
    for i in range(1, len(coords)):
        d = _haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
        cumulative += d
        road.append((coords[i][0], coords[i][1], cumulative))
    return road


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


# --- config ---


def _reload_config(ctx: AciesContext) -> None:
    """Read the toml config and update app state derived from it."""
    with open(ctx.cfg['config_path'], 'rb') as f:
        config = tomllib.load(f)

    road_coords: list[list[float]] = config['road']['coordinates']
    road = _build_road(road_coords)
    ctx.app['road'] = road
    ctx.app['is_loop'] = _is_loop(road)
    logger.info('road: %d segments, %.1f m total, loop=%s', len(road) - 1, road[-1][2], ctx.app['is_loop'])

    ctx.app['node_mapping'] = config.get('map_node_mapping', {})

    tracker_cfg: dict[str, Any] = config.get('tracker', {})
    ctx.app['speed'] = float(tracker_cfg.get('speed', 5.0))  # m/s
    ctx.app['direction'] = int(tracker_cfg.get('direction', 1))  # +1 or -1

    ctx.app['config_mtime'] = os.path.getmtime(ctx.cfg['config_path'])
    logger.info('tracker speed=%.1f m/s direction=%d', ctx.app['speed'], ctx.app['direction'])


# --- app ---


@app.on_startup
def setup(ctx: AciesContext) -> None:
    _reload_config(ctx)

    ensemble_win = ctx.cfg.get('ensemble_win', 30)
    ctx.app['predictions'] = TimeWindow(window_ns=ensemble_win * _NS_PER_S, data_clock=True)
    ctx.app['energy'] = TimeWindow(window_ns=10 * _NS_PER_S, data_clock=True)

    # start at beginning of road (arc=0 for direction=+1, arc=total for direction=-1)
    road: Road = ctx.app['road']
    ctx.app['ref_arc'] = 0.0 if ctx.app['direction'] == 1 else road[-1][2]
    ctx.app['ref_time_ns'] = 0


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

    host, _, rest = source.partition('/')
    node_mapping: dict[str, str] = ctx.app['node_mapping']
    canonical = node_mapping.get(host, host)
    cannoical_topic = f'{canonical}/{rest}'

    energy_win: TimeWindow = ctx.app['energy']
    energy_win.add(cannoical_topic, ts_ns, energy)
    # logger.debug('energy: %8s=%4.0f', cannoical_topic, energy)


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

    # energy_win: TimeWindow = ctx.app['energy']
    logger.debug('>>>> t=%s', int(energy_win.latest_ts / _NS_PER_S))
    logger.debug('>>>> %s', energy_win.keys())
    for k in sorted(energy_win.keys()):
        logger.debug('>>>> %8s: %s', k, ' '.join([f'{x[1]:7.2f}' for x in energy_win.get(k)]))

    road: Road = ctx.app['road']
    total = road[-1][2]
    is_loop: bool = ctx.app['is_loop']

    # set reference time on first data
    if ctx.app['ref_time_ns'] == 0:
        ctx.app['ref_time_ns'] = now_ns

    speed: float = ctx.app['speed'] * ctx.app['direction']
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
