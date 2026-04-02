"""Replay sensor node for AciesOS.

Replays mic or geo data from parquet files, publishing 1-second
AciesTimeSeries windows at real-time (or scaled) pace.

Data format (parquet columns):
  timestamp          float64   corrected epoch seconds
  samples            int64     one ADC sample value
  channel            str|int   channel identifier (e.g. 'EH3' or 0)
  original_timestamp float64   raw capture timestamp

File naming convention:
  {data_dir}/{scene}/run{run}_{node}_{modality}.parquet

Synchronization:
  Pass the same --start-at epoch timestamp to multiple replay nodes.
  Each node loads data, sleeps until that wallclock, then begins.
  With NTP-synced clocks, all nodes start within ms of each other.

Usage::

    acies-replay --data-dir packages/sensors/data \
                 --scene 2024-08-gq \
                 --node gq-1 --run 50 --modality geo \
                 [--speed 1.0] [--start-at EPOCH] [--loop] \
                 [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import threading
import time

import click
import numpy as np
import polars as pl
from acies.corev2 import AciesApp, AciesContext, OnChange, setup_logging
from acies.corev2.msg import AciesKvChange, AciesTimeSeries

logger = logging.getLogger(__name__)

app = AciesApp()

# --- sampling rates by modality ---
_SAMPLING_RATES: dict[str, int] = {
    'geo': 200,
    'mic': 16000,
}

# --- numpy dtypes matching the live sensors ---
_DTYPES: dict[str, str] = {
    'geo': 'int32',
    'mic': 'int16',
}


def _load_windows(path: str, modality: str) -> list[tuple[int, list[bytes], list[str], int, str]]:
    """Load a parquet file and group samples into 1-second windows.

    Returns a list of (timestamp_ns, payload_list, channel_list, sampling_rate, dtype)
    tuples, one per window, sorted by timestamp.
    """
    df = pl.read_parquet(path)
    sampling_rate = _SAMPLING_RATES[modality]
    dtype = _DTYPES[modality]
    np_dtype = np.dtype(dtype)

    # Floor timestamp to integer seconds to group into 1-second windows
    df = df.with_columns(pl.col('timestamp').floor().cast(pl.Int64).alias('window_ts'))

    # Channel column may be str (geo) or int (mic) -- normalize to str
    df = df.with_columns(pl.col('channel').cast(pl.String).alias('channel_str'))

    windows: list[tuple[int, list[bytes], list[str], int, str]] = []
    for (window_ts,), group in df.group_by(['window_ts'], maintain_order=True):
        ts_ns = int(window_ts) * 1_000_000_000  # type: ignore[arg-type]
        channels: list[str] = []
        payloads: list[bytes] = []
        for (ch,), ch_group in group.group_by(['channel_str'], maintain_order=True):
            channels.append(str(ch))
            samples = ch_group['samples'].to_numpy().astype(np_dtype)
            payloads.append(samples.tobytes())
        windows.append((ts_ns, payloads, channels, sampling_rate, dtype))

    windows.sort(key=lambda w: w[0])
    logger.info(
        'loaded %s: %d windows (%.0fs), %d total samples',
        path,
        len(windows),
        len(windows),
        df.height,
    )
    return windows


@app.on_startup
def setup(ctx: AciesContext) -> None:
    data_dir: str = ctx.cfg['data_dir']
    scene: str = ctx.cfg['scene']
    node: str = ctx.cfg['node']
    run: int = ctx.cfg['run']
    modality: str = ctx.cfg['modality']

    path = f'{data_dir}/{scene}/run{run}_{node}_{modality}.parquet'
    windows = _load_windows(path, modality)
    if not windows:
        logger.error('no data in %s', path)
        raise SystemExit(1)

    ctx.app['windows'] = windows
    ctx.app['topic'] = ctx.cfg.get('topic') or ctx.ns.base
    ctx.app['speed'] = ctx.cfg.get('speed', 1.0)
    ctx.app['loop'] = ctx.cfg.get('loop', False)
    ctx.app['start_at'] = ctx.cfg.get('start_at')
    ctx.app['restart'] = threading.Event()
    ctx.app['reload'] = threading.Event()


# --- kv change notifications ---
# Data params: reload windows and restart replay.
_DATA_KEYS = frozenset({'scene', 'node', 'run', 'modality', 'data_dir'})


@app.subscribe(OnChange('scene'), OnChange('node'), OnChange('run'), OnChange('modality'), OnChange('data_dir'))
def on_data_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('data config changed: %s %s', msg.op, msg.key)
    ctx.app['reload'].set()
    ctx.app['restart'].set()


@app.subscribe(OnChange('start_at'))
def on_start_at_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('start_at changed to %s', msg.value)
    ctx.app['restart'].set()


@app.subscribe(OnChange('speed'))
def on_speed_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('speed changed to %s', msg.value)
    ctx.app['restart'].set()


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('replay stopped')


def _wait_for_any(events: list[threading.Event], timeout: float) -> bool:
    """Wait until any event in the list is set, or timeout expires.

    Returns True if any event was set, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        for e in events:
            if e.is_set():
                return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        # Poll in short intervals so we notice any event promptly
        _ = events[0].wait(timeout=min(remaining, 0.1))


def _try_reload(ctx: AciesContext) -> bool:
    """Reload windows from config. Returns True on success."""
    data_dir = ctx.cfg['data_dir']
    scene = ctx.cfg['scene']
    node = ctx.cfg['node']
    run = ctx.cfg['run']
    modality = ctx.cfg['modality']
    path = f'{data_dir}/{scene}/run{run}_{node}_{modality}.parquet'
    try:
        windows = _load_windows(path, modality)
    except Exception:
        logger.exception('failed to reload from %s', path)
        return False
    if not windows:
        logger.error('no data in %s', path)
        return False
    ctx.app['windows'] = windows
    return True


def _play_windows(
    ctx: AciesContext,
    cancel: list[threading.Event],
    windows: list[tuple[int, list[bytes], list[str], int, str]],
    topic: str,
    speed: float,
) -> bool:
    """Play all windows at the given speed. Returns True if completed, False if interrupted."""
    wall_start = time.monotonic()
    data_start_ns = windows[0][0]
    for ts_ns, payloads, channels, sampling_rate, dtype in windows:
        if any(e.is_set() for e in cancel):
            return False
        data_elapsed_s = (ts_ns - data_start_ns) / 1_000_000_000
        wall_target = wall_start + data_elapsed_s / speed
        sleep_s = wall_target - time.monotonic()
        if sleep_s > 0 and _wait_for_any(cancel, timeout=sleep_s):
            return False
        ctx.publish(
            topic,
            AciesTimeSeries(
                source=ctx.ns.base,
                timestamp=ts_ns,
                payload=payloads,
                channels=channels,
                sampling_rate=sampling_rate,
                dtype=dtype,
            ),
        )
        logger.debug('published window t=%d (%d channel(s))', ts_ns, len(channels))
    return True


@app.thread
def replay(ctx: AciesContext, stop: threading.Event) -> None:
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

        windows = ctx.app['windows']
        topic: str = ctx.cfg.get('topic') or ctx.ns.base
        speed: float = ctx.cfg.get('speed', 1.0)
        loop: bool = ctx.cfg.get('loop', False)
        start_at: float | None = ctx.cfg.get('start_at')

        # --- wait for start_at if specified ---
        if start_at is not None:
            delay = start_at - time.time()
            if delay > 0:
                logger.info('waiting %.1fs until start_at=%.3f', delay, start_at)
                if _wait_for_any(cancel, timeout=delay):
                    if stop.is_set():
                        return
                    continue
            else:
                logger.warning('start_at is %.1fs in the past; starting immediately', -delay)

        completed = _play_windows(ctx, cancel, windows, topic, speed)
        if stop.is_set():
            return
        if not completed:
            logger.info('replay interrupted by config change')
            continue

        total_s = (windows[-1][0] - windows[0][0]) / 1_000_000_000
        logger.info('replay complete: %d windows, %.0fs of data at %.1fx speed', len(windows), total_s, speed)

        if not loop:
            logger.info('not looping; shutting down')
            app.stop()
            return
        logger.info('looping replay')


@app.cli()
@click.option('--data-dir', required=True, type=click.Path(exists=True, file_okay=False), help='Root data directory.')
@click.option('--scene', required=True, help='Scene subdirectory (e.g. 2024-08-gq).')
@click.option('--node', required=True, help='Node name in filename (e.g. gq-1).')
@click.option('--run', required=True, type=int, help='Run ID (e.g. 50).')
@click.option(
    '--modality',
    required=True,
    type=click.Choice(['geo', 'mic']),
    help='Sensor modality.',
)
@click.option('--speed', default=1.0, type=float, show_default=True, help='Replay speed multiplier.')
@click.option('--start-at', default=None, type=float, help='Start wallclock as epoch timestamp.')
@click.option('--loop/--no-loop', default=False, show_default=True, help='Loop replay indefinitely.')
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
def main(
    data_dir: str,
    scene: str,
    node: str,
    run: int,
    modality: str,
    speed: float,
    start_at: float | None,
    loop: bool,
    topic: str | None,
) -> None:
    app.state.config.update(
        {
            'data_dir': data_dir,
            'scene': scene,
            'node': node,
            'run': run,
            'modality': modality,
            'speed': speed,
            'start_at': start_at,
            'loop': loop,
            'topic': topic,
        }
    )
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
