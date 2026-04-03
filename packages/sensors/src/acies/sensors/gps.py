"""GPS sensor node for AciesOS.

Publishes GPS positions. Source is inferred from CLI arguments:
  --file       replay mode (parquet with time, latitude, longitude, elevation)
  --broker     MQTT mode (live GPS from MQTT broker, TODO)

Publishes a dict per tick: {'lat': float, 'lon': float, 'elevation': float}

Usage::

    acies-gps --file packages/sensors/data/run2_gps.parquet \
              [--speed 1.0] [--start-at EPOCH] [--loop] [--topic ws://gps] \
              [--acies-host HOST] [--acies-name NAME]

    acies-gps --broker 192.168.70.51 --mqtt-topic '/+/gps' \
              [--topic ws://gps] [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import click
import polars as pl
from acies.corev2 import AciesApp, AciesContext, setup_logging

logger = logging.getLogger(__name__)

app = AciesApp()

_NS_PER_S = 1_000_000_000

# Filename pattern: run<id>_<label>_gps.parquet or run<id>_gps.parquet
_GPS_FILENAME_RE = re.compile(r'run\d+_(?:(.+)_)?gps\.parquet$')


def _extract_label(path: str) -> str | None:
    """Extract the target label from a GPS parquet filename.

    run59_polaris_gps.parquet -> 'polaris'
    run2_gps.parquet          -> None
    """
    m = _GPS_FILENAME_RE.search(Path(path).name)
    if m:
        return m.group(1)
    return None


def _load_gps(path: str) -> list[tuple[int, dict[str, float]]]:
    """Load a GPS parquet file.

    Expects columns: time (datetime or numeric), latitude, longitude, elevation.
    Returns a sorted list of (timestamp_ns, position_dict) tuples.
    """
    df = pl.read_parquet(path)

    # Convert time column to nanosecond timestamps
    if df['time'].dtype.is_(pl.Datetime) or isinstance(df['time'].dtype, pl.Datetime):
        # Polars Datetime cast to Int64 gives microseconds — convert to ns
        ts_us = df['time'].cast(pl.Int64).to_list()
        ts_ns = [int(t * 1000) for t in ts_us]
    else:
        # Numeric — detect unit by magnitude
        raw = df['time'].cast(pl.Float64).to_list()
        if raw[0] > 1e17:
            ts_ns = [int(t) for t in raw]  # nanoseconds
        elif raw[0] > 1e14:
            ts_ns = [int(t * 1_000) for t in raw]  # microseconds
        elif raw[0] > 1e11:
            ts_ns = [int(t * 1_000_000) for t in raw]  # milliseconds
        else:
            ts_ns = [int(t * _NS_PER_S) for t in raw]  # seconds

    lats = df['latitude'].to_list()
    lons = df['longitude'].to_list()
    elevs = df['elevation'].to_list() if 'elevation' in df.columns else [0.0] * df.height

    positions: list[tuple[int, dict[str, float]]] = [
        (ts_ns[i], {'lat': lats[i], 'lon': lons[i], 'elevation': elevs[i]}) for i in range(df.height)
    ]
    positions.sort(key=lambda p: p[0])
    logger.info('loaded %s: %d positions (%.0fs)', path, len(positions), len(positions))
    return positions


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


def _infer_mode(cfg: dict[str, Any]) -> str:
    """Infer GPS mode from CLI arguments."""
    if cfg.get('file'):
        return 'replay'
    if cfg.get('broker'):
        return 'mqtt'
    raise ValueError('provide --file for replay or --broker for MQTT')


@app.on_startup
def setup(ctx: AciesContext) -> None:
    mode = _infer_mode(ctx.cfg)
    ctx.app['mode'] = mode
    ctx.app['topic'] = ctx.cfg.get('topic') or ctx.ns.topic('gps')

    if mode == 'replay':
        path: str = ctx.cfg['file']
        positions = _load_gps(path)
        if not positions:
            logger.error('no GPS data in %s', path)
            raise SystemExit(1)
        label = ctx.cfg.get('label') or _extract_label(path) or 'unknown'
        ctx.app['positions'] = positions
        ctx.app['label'] = label
        ctx.app['speed'] = ctx.cfg.get('speed', 1.0)
        ctx.app['loop'] = ctx.cfg.get('loop', False)
        ctx.app['start_at'] = ctx.cfg.get('start_at')
        logger.info('gps replay: label=%r, %d positions', label, len(positions))
    else:
        logger.error('MQTT GPS source not yet implemented')
        raise SystemExit(1)


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('gps stopped')


@app.thread
def gps_replay(ctx: AciesContext, stop: threading.Event) -> None:
    if ctx.app['mode'] != 'replay':
        return

    positions: list[tuple[int, dict[str, float]]] = ctx.app['positions']
    label: str = ctx.app['label']
    topic: str = ctx.app['topic']
    speed: float = ctx.app['speed']
    loop: bool = ctx.app['loop']
    start_at: float | None = ctx.app['start_at']
    cancel = [stop]

    if start_at is not None:
        delay = start_at - time.time()
        if delay > 0:
            logger.info('waiting %.1fs until start_at=%.3f', delay, start_at)
            if _wait_for_any(cancel, timeout=delay):
                return
        else:
            logger.warning('start_at is %.1fs in the past; starting immediately', -delay)

    while not stop.is_set():
        wall_start = time.monotonic()
        data_start_ns = positions[0][0]

        for ts_ns, pos in positions:
            if stop.is_set():
                return
            data_elapsed_s = (ts_ns - data_start_ns) / _NS_PER_S
            wall_target = wall_start + data_elapsed_s / speed
            sleep_s = wall_target - time.monotonic()
            if sleep_s > 0:
                if _wait_for_any(cancel, timeout=sleep_s):
                    return
            ctx.publish(topic, {label: pos})
            logger.debug('gps %s: lat=%.6f lon=%.6f', label, pos['lat'], pos['lon'])

        total_s = (positions[-1][0] - data_start_ns) / _NS_PER_S
        logger.info('gps replay complete: %d positions, %.0fs at %.1fx', len(positions), total_s, speed)

        if not loop:
            logger.info('not looping; shutting down')
            app.stop()
            return
        logger.info('looping gps replay')


@app.cli()
@click.option(
    '--file', default=None, type=click.Path(exists=True, dir_okay=False), help='GPS parquet file (replay mode).'
)
@click.option('--label', default=None, help='Target label. Auto-extracted from filename if omitted.')
@click.option('--speed', default=1.0, type=float, show_default=True, help='Replay speed multiplier.')
@click.option('--start-at', default=None, type=float, help='Start wallclock as epoch timestamp.')
@click.option('--loop/--no-loop', default=False, show_default=True, help='Loop replay indefinitely.')
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
@click.option('--broker', default=None, help='MQTT broker address (live mode).')
@click.option('--mqtt-topic', default=None, help='MQTT topic to subscribe to (live mode).')
def main(**kwargs: Any) -> None:
    if not kwargs.get('file') and not kwargs.get('broker'):
        raise click.UsageError('provide --file for replay or --broker for MQTT')
    if kwargs.get('file') and kwargs.get('broker'):
        raise click.UsageError('--file and --broker are mutually exclusive')
    app.state.config.update(kwargs)
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
