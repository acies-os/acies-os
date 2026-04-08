"""GPS sensor node for AciesOS.

Publishes GPS positions. Source is inferred from CLI arguments:
  --data-dir   replay mode (parquet with time, latitude, longitude, elevation)
  --broker     MQTT mode (live GPS from MQTT broker, TODO)

File naming convention:
  {data_dir}/{scene}/run{run}_{label}_gps.parquet

Publishes a dict per tick: {label: {'lat': float, 'lon': float, 'elevation': float}}

Usage::

    acies-gps --data-dir packages/sensors/data --scene 2024-03-29-ICT \
              --run 2 --label miata \
              [--speed 1.0] [--start-at EPOCH] [--loop] \
              [--acies-namespace NS] [--acies-name NAME]

    acies-gps --broker 192.168.70.51 --mqtt-topic '/+/gps' \
              [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import click
import polars as pl
from acies.corev2 import AciesApp, AciesContext, OnChange, setup_logging
from acies.corev2.msg import AciesKvChange

logger = logging.getLogger(__name__)

app = AciesApp()

_NS_PER_S = 1_000_000_000


# --- parquet loading ---


def _build_path(ctx: AciesContext) -> str:
    """Build GPS parquet path from config keys."""
    return f'{ctx.cfg["data_dir"]}/{ctx.cfg["scene"]}/run{ctx.cfg["run"]}_{ctx.cfg["label"]}_gps.parquet'


def _load_gps(path: str) -> list[tuple[int, dict[str, float]]]:
    """Load a GPS parquet file.

    Expects columns: time (datetime or numeric), latitude, longitude, elevation.
    Returns a sorted list of (timestamp_ns, position_dict) tuples.
    """
    df = pl.read_parquet(path)

    # Convert time column to nanosecond timestamps
    if df['time'].dtype.is_(pl.Datetime) or isinstance(df['time'].dtype, pl.Datetime):
        ts_us = df['time'].cast(pl.Int64).to_list()
        ts_ns = [int(t * 1000) for t in ts_us]
    else:
        raw = df['time'].cast(pl.Float64).to_list()
        if raw[0] > 1e17:
            ts_ns = [int(t) for t in raw]
        elif raw[0] > 1e14:
            ts_ns = [int(t * 1_000) for t in raw]
        elif raw[0] > 1e11:
            ts_ns = [int(t * 1_000_000) for t in raw]
        else:
            ts_ns = [int(t * _NS_PER_S) for t in raw]

    lats = df['latitude'].to_list()
    lons = df['longitude'].to_list()
    elevs = df['elevation'].to_list() if 'elevation' in df.columns else [0.0] * df.height

    positions: list[tuple[int, dict[str, float]]] = [
        (ts_ns[i], {'lat': lats[i], 'lon': lons[i], 'elevation': elevs[i]}) for i in range(df.height)
    ]
    positions.sort(key=lambda p: p[0])
    logger.info('loaded %s: %d positions (%.0fs)', path, len(positions), len(positions))
    return positions


# --- event helpers ---


def _wait_for_any(events: list[threading.Event], timeout: float) -> bool:
    """Wait until any event is set or timeout expires. Returns True if any was set."""
    deadline = time.monotonic() + timeout
    while True:
        for e in events:
            if e.is_set():
                return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        _ = events[0].wait(timeout=min(remaining, 0.1))


def _sleep_until(start_at: float, cancel: list[threading.Event]) -> bool:
    """Sleep until the given wallclock epoch. Returns True if ready, False if interrupted."""
    delay = start_at - time.time()
    if delay > 0:
        logger.info('waiting %.1fs until start_at=%.3f', delay, start_at)
        if _wait_for_any(cancel, timeout=delay):
            return False
    else:
        logger.warning('start_at is %.1fs in the past; starting immediately', -delay)
    return True


def _wait_for_start_at(ctx: AciesContext, cancel: list[threading.Event]) -> bool:
    """Wait until a new start_at is available, then sleep until that wallclock.

    Returns True if ready to play, False if interrupted.
    """
    start_at_ev: threading.Event = ctx.app['start_at_ev']
    wake = [*cancel, start_at_ev]

    while not any(e.is_set() for e in cancel):
        if start_at_ev.is_set():
            start_at: float | None = ctx.cfg.get('start_at')
            if start_at is not None:
                start_at_ev.clear()
                ctx.app['last_start_at'] = start_at
                break
        logger.debug('waiting for start_at...')
        if _wait_for_any(wake, timeout=1.0):
            if any(e.is_set() for e in cancel):
                return False
            # woken by start_at_ev -- loop to consume it
    else:
        return False

    return _sleep_until(start_at, cancel)


def _await_start(ctx: AciesContext, cancel: list[threading.Event], start_at: float | None) -> bool:
    """Handle start_at logic. Returns True when ready to play, False to re-loop."""
    first_run = ctx.app['last_start_at'] is None
    if first_run and start_at is not None:
        ctx.app['last_start_at'] = start_at
        return _sleep_until(start_at, cancel)
    if first_run:
        return True  # no start_at on CLI -> play immediately
    logger.info('data loaded, waiting for start_at to begin playback')
    return _wait_for_start_at(ctx, cancel)


def _try_reload(ctx: AciesContext) -> bool:
    """Reload GPS positions from config. Returns True on success."""
    path = _build_path(ctx)
    try:
        logger.debug('loading GPS from %s', path)
        positions = _load_gps(path)
    except Exception:
        logger.exception('failed to reload from %s', path)
        return False
    if not positions:
        logger.error('no GPS data in %s', path)
        return False
    ctx.app['positions'] = positions
    return True


def _play_positions(
    ctx: AciesContext,
    cancel: list[threading.Event],
    positions: list[tuple[int, dict[str, float]]],
    label: str,
    topic: str,
    speed: float,
) -> bool:
    """Play all positions at the given speed. Returns True if completed, False if interrupted."""
    wall_start = time.monotonic()
    data_start_ns = positions[0][0]
    for ts_ns, pos in positions:
        if any(e.is_set() for e in cancel):
            return False
        data_elapsed_s = (ts_ns - data_start_ns) / _NS_PER_S
        wall_target = wall_start + data_elapsed_s / speed
        sleep_s = wall_target - time.monotonic()
        if sleep_s > 0 and _wait_for_any(cancel, timeout=sleep_s):
            return False
        ctx.publish(topic, {label: pos})
        logger.debug('%s <- %s: lat=%.6f lon=%.6f', topic, label, pos['lat'], pos['lon'])
    return True


# --- lifecycle ---


@app.on_startup
def setup(ctx: AciesContext) -> None:
    mode = 'replay' if ctx.cfg.get('data_dir') else 'mqtt'
    ctx.app['mode'] = mode
    ctx.cfg['topic'] = ctx.cfg.get('topic') or ctx.ns.topic('truth')

    if mode == 'replay':
        path = _build_path(ctx)
        positions = _load_gps(path)
        if not positions:
            logger.error('no GPS data in %s', path)
            raise SystemExit(1)
        ctx.app['positions'] = positions
        ctx.app['restart'] = threading.Event()
        ctx.app['reload'] = threading.Event()
        ctx.app['start_at_ev'] = threading.Event()
        ctx.app['last_start_at'] = None
        logger.info('gps replay: label=%r, %d positions', ctx.cfg['label'], len(positions))
    else:
        logger.error('MQTT GPS source not yet implemented')
        raise SystemExit(1)


# --- kv change notifications ---


@app.subscribe(OnChange('scene'), OnChange('run'), OnChange('label'))
def on_data_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('gps data config changed: %s %s', msg.op, msg.key)
    ctx.app['reload'].set()
    ctx.app['restart'].set()


@app.subscribe(OnChange('start_at'))
def on_start_at_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('start_at changed to %s', msg.value)
    ctx.app['start_at_ev'].set()
    ctx.app['restart'].set()


@app.subscribe(OnChange('speed'))
def on_speed_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('speed changed to %s', msg.value)
    ctx.app['restart'].set()


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('gps stopped')


# --- replay thread ---


@app.thread
def gps_replay(ctx: AciesContext, stop: threading.Event) -> None:
    if ctx.app['mode'] != 'replay':
        return

    restart: threading.Event = ctx.app['restart']
    reload_ev: threading.Event = ctx.app['reload']
    cancel = [stop, restart]

    while not stop.is_set():
        restart.clear()

        if reload_ev.is_set():
            reload_ev.clear()
            if not _try_reload(ctx):
                _ = _wait_for_any(cancel, timeout=3600)
                continue

        positions = ctx.app['positions']
        label: str = ctx.cfg['label']
        topic: str = ctx.cfg['topic']
        speed: float = ctx.cfg.get('speed', 1.0)
        loop: bool = ctx.cfg.get('loop', False)
        start_at: float | None = ctx.cfg.get('start_at')

        # --- determine when to start playback ---
        ready = _await_start(ctx, cancel, start_at)
        if stop.is_set():
            return
        if not ready:
            continue

        completed = _play_positions(ctx, cancel, positions, label, topic, speed)
        if stop.is_set():
            return
        if not completed:
            logger.info('gps replay interrupted by config change')
            continue

        total_s = (positions[-1][0] - positions[0][0]) / _NS_PER_S
        logger.info('gps replay complete: %d positions, %.0fs at %.1fx', len(positions), total_s, speed)

        if not loop:
            logger.info('not looping; shutting down')
            app.stop()
            return
        logger.info('looping gps replay')


@app.cli()
@click.option(
    '--data-dir',
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help='Root data directory (replay mode).',
)
@click.option('--scene', default=None, help='Scene subdirectory (e.g. 2024-03-29-ICT).')
@click.option('--run', default=None, type=int, help='Run ID (e.g. 2).')
@click.option('--label', default=None, help='Target label (e.g. miata).')
@click.option('--speed', default=1.0, type=float, show_default=True, help='Replay speed multiplier.')
@click.option('--start-at', default=None, type=float, help='Start wallclock as epoch timestamp.')
@click.option('--loop/--no-loop', default=False, show_default=True, help='Loop replay indefinitely.')
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
@click.option('--broker', default=None, help='MQTT broker address (live mode).')
@click.option('--mqtt-topic', default=None, help='MQTT topic to subscribe to (live mode).')
def main(**kwargs: Any) -> None:
    has_replay = kwargs.get('data_dir')
    has_mqtt = kwargs.get('broker')
    if not has_replay and not has_mqtt:
        raise click.UsageError('provide --data-dir for replay or --broker for MQTT')
    if has_replay and has_mqtt:
        raise click.UsageError('--data-dir and --broker are mutually exclusive')
    if has_replay:
        for key in ('scene', 'run', 'label'):
            if kwargs.get(key) is None:
                raise click.UsageError(f'--{key} is required for replay mode')
    app.state.config.update(kwargs)
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
