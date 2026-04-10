"""Vehicle position tracker for AciesOS.

Uses a particle filter with a constant-velocity motion model to
estimate the vehicle's position and speed along a road.

The road is modelled as a polyline projected into 1-D arc-length.
Each sensor node has a known arc position. The measurement model
uses relative log-energy ratios: closer sensors see higher energy.
A log-distance attenuation model relates particle-to-sensor arc
distance to expected relative energy.

Inputs:
    **/energy           energy dict from sensor nodes
    **/vehicle          classifier predictions (for vehicle label)
    config [road]       road polyline coordinates
    config [gps]        sensor node GPS coordinates
    config [tracker]    particle filter parameters

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
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import click
import numpy as np
import numpy.typing as npt
import tomli as tomllib
from acies.buffers.temporal import TimeWindow
from acies.core import AciesApp, AciesContext, OnChange, setup_logging
from acies.core.msg import AciesInference, AciesKvChange
from numpy.random import Generator

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


# --- particle filter ---


def systematic_resample(weights: npt.NDArray[np.float64], rng: Generator | None = None) -> npt.NDArray[np.intp]:
    """Systematic resampling. Returns indices of selected particles."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    return np.searchsorted(cumsum, positions)


@dataclass
class RoadParticleFilter:
    """Particle filter for 1-D road tracking with 2-D measurement model.

    State per particle: (s, v) where s is arc position in metres
    and v is speed in m/s (signed: positive = increasing arc).

    The measurement model uses physical (Euclidean) distances in XY
    space, not arc distances. This is important for loop roads where
    sensors on opposite sides of the loop are physically close but
    far apart in arc space.

    For each particle, arc position is converted to XY via road polyline
    interpolation, then Euclidean distance to each sensor is computed.
    Relative log-energy ratios (sensor 0 as reference) are compared
    against observations.
    """

    n_particles: int
    dt: float  # seconds between predict steps
    sigma_s: float  # position process noise std (metres)
    sigma_v: float  # velocity process noise std (m/s)
    road_length: float
    road_arcs: npt.NDArray[np.float64]  # shape (P,) — arc at each road polyline point
    road_xy: npt.NDArray[np.float64]  # shape (P, 2) — XY at each road polyline point
    sensor_xy: npt.NDArray[np.float64]  # shape (M, 2) — XY positions of sensors
    eta: float = 1.0  # attenuation exponent in log-distance model
    d0: float = 1.0  # distance floor to avoid log(0)
    meas_sigma: float = 1.0  # measurement noise std in relative log-energy space
    is_loop: bool = False
    v_min: float = -30.0
    v_max: float = 30.0

    # --- particle state (initialized in __post_init__) ---
    s: npt.NDArray[np.float64] = field(init=False)
    v: npt.NDArray[np.float64] = field(init=False)
    w: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.road_arcs = np.asarray(self.road_arcs, dtype=np.float64)
        self.road_xy = np.asarray(self.road_xy, dtype=np.float64)
        self.sensor_xy = np.asarray(self.sensor_xy, dtype=np.float64)
        self.s = np.zeros(self.n_particles, dtype=np.float64)
        self.v = np.zeros(self.n_particles, dtype=np.float64)
        self.w = np.ones(self.n_particles, dtype=np.float64) / self.n_particles

    # --- arc to XY ---

    def _arc_to_xy(self, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Interpolate arc positions to XY coordinates along the road polyline.

        Args:
            s: shape (N,) — arc positions of particles.

        Returns shape (N, 2) — XY coordinates.
        """
        # clamp/wrap arc values
        if self.is_loop:
            s = s % self.road_length
        else:
            s = np.clip(s, 0.0, self.road_length)

        # find segment index for each particle
        idx = np.searchsorted(self.road_arcs, s, side='right') - 1
        idx = np.clip(idx, 0, len(self.road_arcs) - 2)

        # interpolation parameter within each segment
        seg_start = self.road_arcs[idx]
        seg_end = self.road_arcs[idx + 1]
        seg_len = seg_end - seg_start
        # avoid division by zero for zero-length segments
        safe_len = np.where(seg_len > 0, seg_len, 1.0)
        t = (s - seg_start) / safe_len
        t = np.clip(t, 0.0, 1.0)

        xy_start = self.road_xy[idx]  # (N, 2)
        xy_end = self.road_xy[idx + 1]  # (N, 2)
        return xy_start + t[:, None] * (xy_end - xy_start)

    # --- initialization ---

    def initialize_uniform(self, rng: Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.s = rng.uniform(0.0, self.road_length, size=self.n_particles)
        self.v = rng.uniform(self.v_min, self.v_max, size=self.n_particles)
        self.w[:] = 1.0 / self.n_particles

    # --- predict ---

    def predict(self, rng: Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.v = np.clip(
            self.v + rng.normal(0.0, self.sigma_v, size=self.n_particles),
            self.v_min,
            self.v_max,
        )
        self.s = self.s + self.v * self.dt + rng.normal(0.0, self.sigma_s, size=self.n_particles)
        if self.is_loop:
            self.s %= self.road_length
        else:
            self.s = np.clip(self.s, 0.0, self.road_length)

    # --- measurement model ---

    def _expected_relative_log_energy(self, s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Expected relative log-energy for each particle.

        Converts arc -> XY, computes Euclidean distance to each sensor,
        then computes relative log-energy with sensor 0 as reference.
        Returns shape (N, M-1).
        """
        xy = self._arc_to_xy(s)  # (N, 2)
        # Euclidean distance from each particle to each sensor: (N, M)
        diff = xy[:, None, :] - self.sensor_xy[None, :, :]  # (N, M, 2)
        dists = np.linalg.norm(diff, axis=2)  # (N, M)
        ref = dists[:, 0:1]  # (N, 1)
        return -self.eta * np.log((dists[:, 1:] + self.d0) / (ref + self.d0))

    @staticmethod
    def _observed_relative_log_energy(
        raw_energy: npt.NDArray[np.float64], eps: float = 1e-12
    ) -> npt.NDArray[np.float64]:
        """Convert raw sensor energies (M,) to relative log-energy (M-1,)."""
        loge = np.log(raw_energy + eps)
        return loge[1:] - loge[0]

    # --- update ---

    def update(self, raw_energy: npt.NDArray[np.float64]) -> None:
        """Update weights using measured raw energies from all sensors."""
        z = self._observed_relative_log_energy(raw_energy)  # (M-1,)
        z_hat = self._expected_relative_log_energy(self.s)  # (N, M-1)
        residual = z_hat - z[None, :]  # (N, M-1)

        # Gaussian log-likelihood
        ll = -0.5 * np.sum((residual / self.meas_sigma) ** 2, axis=1)
        ll -= np.max(ll)  # numerical stability
        w_new = np.exp(ll)

        self.w *= w_new
        self.w += 1e-300
        self.w /= np.sum(self.w)

    # --- resampling ---

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.w**2))

    def resample_if_needed(self, threshold_ratio: float = 0.5, rng: Generator | None = None) -> None:
        if self.effective_sample_size() < threshold_ratio * self.n_particles:
            idx = systematic_resample(self.w, rng=rng)
            self.s = self.s[idx]
            self.v = self.v[idx]
            self.w[:] = 1.0 / self.n_particles

    # --- estimation ---

    def estimate(self) -> tuple[float, float]:
        """Weighted mean estimate of (arc_position, speed)."""
        return float(np.sum(self.w * self.s)), float(np.sum(self.w * self.v))


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


def _latlon_to_xy(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert lat/lon to local XY in metres, using a flat-Earth approximation.

    Args:
        lat, lon: point to convert.
        ref_lat, ref_lon: reference origin (XY = 0, 0).

    Returns (x_metres, y_metres) where x is east and y is north.
    """
    cos_ref = math.cos(ref_lat * _DEG_TO_RAD)
    x = (lon - ref_lon) * _DEG_TO_RAD * _EARTH_R * cos_ref
    y = (lat - ref_lat) * _DEG_TO_RAD * _EARTH_R
    return x, y


def _build_road(coords: list[list[float]]) -> Road:
    """Convert a polyline of [lat, lon] into (lat, lon, cumulative_distance_m)."""
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


def _project_onto_road(road: Road, lat: float, lon: float) -> Metres:
    """Project a lat/lon point onto the road polyline. Returns arc-length in metres."""
    candidates: list[tuple[Metres, Metres]] = []
    for i in range(len(road) - 1):
        lat1, lon1, arc1 = road[i]
        lat2, lon2, arc2 = road[i + 1]
        arc, dist = _project_onto_segment(lat, lon, lat1, lon1, lat2, lon2, arc1, arc2)
        candidates.append((arc, dist))
    if not candidates:
        return 0.0
    return min(candidates, key=lambda c: c[1])[0]


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


def _build_particle_filter(ctx: AciesContext) -> RoadParticleFilter:
    """Create and initialize a particle filter from current app state."""
    road: Road = ctx.app['road']
    is_loop: bool = ctx.app['is_loop']
    sensor_order: list[str] = ctx.app['sensor_order']
    node_xy: dict[str, tuple[float, float]] = ctx.app['node_xy']
    tracker_cfg: dict[str, Any] = ctx.app['tracker_cfg']

    sensor_xy = np.array([node_xy[s] for s in sensor_order], dtype=np.float64)

    pf = RoadParticleFilter(
        n_particles=tracker_cfg.get('n_particles', 500),
        dt=1.0,
        sigma_s=tracker_cfg.get('sigma_s', 2.0),
        sigma_v=tracker_cfg.get('sigma_v', 1.0),
        road_length=road[-1][2],
        road_arcs=ctx.app['road_arcs'],
        road_xy=ctx.app['road_xy'],
        sensor_xy=sensor_xy,
        eta=tracker_cfg.get('eta', 1.0),
        d0=tracker_cfg.get('d0', 5.0),
        meas_sigma=tracker_cfg.get('meas_sigma', 1.0),
        is_loop=is_loop,
        v_min=tracker_cfg.get('v_min', -30.0),
        v_max=tracker_cfg.get('v_max', 30.0),
    )
    pf.initialize_uniform(rng=ctx.app['rng'])
    return pf


def _reload_config(ctx: AciesContext) -> None:
    """Read the toml config and update app state derived from it."""
    with open(ctx.cfg['config_path'], 'rb') as f:
        config = tomllib.load(f)

    road_coords: list[list[float]] = config['road']['coordinates']
    road = _build_road(road_coords)
    ctx.app['road'] = road
    ctx.app['is_loop'] = _is_loop(road)
    logger.info('road: %d segments, %.1f m total, loop=%s', len(road) - 1, road[-1][2], ctx.app['is_loop'])

    # compute local XY for road polyline (reference = first road point)
    ref_lat, ref_lon = road[0][0], road[0][1]
    ctx.app['ref_latlon'] = (ref_lat, ref_lon)
    road_arcs = np.array([p[2] for p in road], dtype=np.float64)
    road_xy = np.array([_latlon_to_xy(p[0], p[1], ref_lat, ref_lon) for p in road], dtype=np.float64)
    ctx.app['road_arcs'] = road_arcs
    ctx.app['road_xy'] = road_xy

    ctx.app['node_mapping'] = config.get('map_node_mapping', {})

    # project sensor nodes onto road and compute their XY
    gps_table: dict[str, list[float]] = config.get('gps', {})
    node_arcs: dict[str, Metres] = {}
    node_xy: dict[str, tuple[float, float]] = {}
    for node, (lat, lon) in gps_table.items():
        node_arcs[node] = _project_onto_road(road, lat, lon)
        node_xy[node] = _latlon_to_xy(lat, lon, ref_lat, ref_lon)
        logger.info(
            'node %s: lat=%.6f lon=%.6f -> arc=%.1f m xy=(%.1f, %.1f)',
            node,
            lat,
            lon,
            node_arcs[node],
            node_xy[node][0],
            node_xy[node][1],
        )
    ctx.app['node_arcs'] = node_arcs
    ctx.app['node_xy'] = node_xy

    # sorted sensor order for consistent energy vector indexing
    ctx.app['sensor_order'] = sorted(node_arcs.keys())
    logger.info('sensor order: %s', ctx.app['sensor_order'])

    ctx.app['tracker_cfg'] = config.get('tracker', {})
    ctx.app['config_mtime'] = os.path.getmtime(ctx.cfg['config_path'])
    logger.info('tracker cfg: %s', ctx.app['tracker_cfg'])


# --- app ---


@app.on_startup
def setup(ctx: AciesContext) -> None:
    ctx.app['rng'] = np.random.default_rng()

    _reload_config(ctx)

    ctx.cfg['start_at'] = None

    ensemble_win = ctx.cfg.get('ensemble_win', 30)
    ctx.app['predictions'] = TimeWindow(window_ns=ensemble_win * _NS_PER_S, data_clock=True)
    ctx.app['energy'] = TimeWindow(window_ns=10 * _NS_PER_S, data_clock=True)

    ctx.app['pf'] = _build_particle_filter(ctx)


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
        ctx.app['pf'] = _build_particle_filter(ctx)


@app.subscribe('**/energy')
def on_energy(ctx: AciesContext, msg: Any) -> None:
    source: str = msg['source']
    ts_ns: int = msg['timestamp']
    energy_by_ch: dict[str, float] = msg['energy']
    energy = max(energy_by_ch.values())

    host, _, rest = source.partition('/')
    node_mapping: dict[str, str] = ctx.app['node_mapping']
    canonical = node_mapping.get(host, host)
    canonical_topic = f'{canonical}/{rest}'

    energy_win: TimeWindow = ctx.app['energy']
    energy_win.add(canonical_topic, ts_ns, energy)
    # logger.debug('energy: %8s=%4.0f', canonical_topic, energy)


@app.subscribe(OnChange('start_at'))
def on_start_at(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('start_at changed to %s; resetting tracker state', msg.value)
    ctx.app['energy'] = TimeWindow(window_ns=10 * _NS_PER_S, data_clock=True)
    ensemble_win = ctx.cfg.get('ensemble_win', 30)
    ctx.app['predictions'] = TimeWindow(window_ns=ensemble_win * _NS_PER_S, data_clock=True)
    ctx.app['pf'] = _build_particle_filter(ctx)


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


def _get_energy_vector(energy_win: TimeWindow, sensor_order: list[str]) -> npt.NDArray[np.float64] | None:
    """Build an energy vector aligned with sensor_order from the latest readings.

    For each sensor, looks for any modality topic (geo or mic) and takes the
    max energy across modalities. Returns None if any sensor has no data.
    """
    energies: list[float] = []
    for sensor in sensor_order:
        # find all topics for this sensor (e.g. rs1/geo, rs1/mic)
        best = 0.0
        found = False
        for key in energy_win.keys():
            if key.startswith(sensor + '/'):
                entry = energy_win.latest(key)
                if entry is not None:
                    _, e = entry
                    best = max(best, e)
                    found = True
        if not found:
            return None
        energies.append(best)
    return np.array(energies, dtype=np.float64)


@app.schedule(1.0)
def estimate(ctx: AciesContext) -> None:
    energy_win: TimeWindow = ctx.app['energy']
    now_ns: int = energy_win.latest_ts
    if now_ns == 0:
        return

    logger.debug('>>>> t=%s', int(energy_win.latest_ts / _NS_PER_S))
    logger.debug('>>>> %s', energy_win.keys())
    for k in sorted(energy_win.keys()):
        logger.debug('>>>> %8s: %s', k, ' '.join([f'{x[1]:7.2f}' for x in energy_win.get(k)]))

    road: Road = ctx.app['road']
    pf: RoadParticleFilter = ctx.app['pf']
    rng: Generator = ctx.app['rng']
    sensor_order: list[str] = ctx.app['sensor_order']

    # --- predict ---
    pf.predict(rng=rng)

    # --- update (if we have energy from all sensors) ---
    energy_vec = _get_energy_vector(energy_win, sensor_order)
    if energy_vec is not None:
        pf.update(energy_vec)
        pf.resample_if_needed(rng=rng)
        logger.debug(
            'pf update: energy=[%s] ESS=%.0f',
            ', '.join(f'{e:.1f}' for e in energy_vec),
            pf.effective_sample_size(),
        )

    # --- estimate ---
    s_hat, v_hat = pf.estimate()
    arc = _wrap_arc(s_hat, road[-1][2], ctx.app['is_loop'])

    label = _ensemble_label(ctx.app['predictions']) or 'unknown'
    lat, lon = _arc_to_latlon(road, arc)
    ctx.publish(ctx.ns.topic('gps'), {label: {'lat': lat, 'lon': lon, 'elevation': 0.0}, 'timestamp': now_ns})
    logger.debug(
        'estimate: arc=%.1f m speed=%.1f m/s lat=%.6f lon=%.6f ESS=%.0f',
        arc,
        v_hat,
        lat,
        lon,
        pf.effective_sample_size(),
    )


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
