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
                 [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import threading
import time

import click
import numpy as np
import numpy.typing as npt
import polars as pl
from acies.corev2 import AciesApp, AciesContext, OnChange, setup_logging
from acies.corev2.msg import AciesKvChange, AciesTimeSeries

logger = logging.getLogger(__name__)

app = AciesApp()


_SAMPLING_RATE = {'geo': 200, 'mic': 16000}
_DTYPE = {'geo': 'int32', 'mic': 'int16'}
_CHANNEL = {'geo': {'SH3', 'EH3'}, 'mic': {'0'}}
_NS_PER_S = 1_000_000_000


def _ts_to_ns(ts: float) -> int:
    """Convert a timestamp to nanoseconds, auto-detecting the unit.

    Handles fractional seconds (~1.7e9), milliseconds (~1.7e12),
    and nanoseconds (~1.7e18).
    """
    if ts > 1e15:
        return int(ts)
    if ts > 1e12:
        return int(ts * 1_000_000)
    return int(ts * _NS_PER_S)


def _load_windows(path: str, modality: str) -> list[tuple[int, list[bytes], list[str], int, str]]:
    """Load a parquet file and chunk samples into 1-second windows.

    Filters to the wanted channels, extracts per-channel sample arrays,
    and slices them in lockstep into ``sampling_rate``-sized chunks.
    Timestamps are assumed identical across channels at each row index.

    Returns a sorted list of (timestamp_ns, payload_list, channel_list,
    sampling_rate, dtype) tuples.
    """
    sampling_rate: int = _SAMPLING_RATE[modality]
    dtype: str = _DTYPE[modality]
    wanted: set[str] = _CHANNEL[modality]
    np_dtype = np.dtype(dtype)

    df = pl.read_parquet(path)

    if 'channel' not in df.columns:
        # No channel column — treat all samples as a single channel
        default_ch = sorted(wanted)[0]
        logger.info('no channel column in %s; assuming single channel %r', path, default_ch)
        present = [default_ch]
        df = df.sort('timestamp')
        ch_arrays: dict[str, npt.NDArray[np.int_]] = {
            default_ch: df['samples'].to_numpy().astype(np_dtype),
        }
        ch_timestamps: list[float] = df['timestamp'].to_list()
        del df
    else:
        df = df.with_columns(pl.col('channel').cast(pl.String).alias('channel_str'))
        available = set(df['channel_str'].unique().to_list())
        present = sorted(wanted & available)
        if not present:
            logger.error('no wanted channels %s in %s; available: %s', wanted, path, sorted(available))
            return []

        df = df.filter(pl.col('channel_str').is_in(present)).sort('timestamp')
        ch_arrays = {}
        ch_timestamps: list[float] | None = None  # type: ignore[assignment]
        for ch in present:
            ch_df = df.filter(pl.col('channel_str') == ch)
            ch_arrays[ch] = ch_df['samples'].to_numpy().astype(np_dtype)
            if ch_timestamps is None:
                ch_timestamps = ch_df['timestamp'].to_list()
            del ch_df
        del df
        assert ch_timestamps is not None

    # All channels should have the same number of samples
    n_samples = len(ch_timestamps)
    for ch, arr in ch_arrays.items():
        if len(arr) != n_samples:
            logger.warning('channel %s has %d samples, expected %d; truncating', ch, len(arr), n_samples)
            ch_arrays[ch] = arr[:n_samples]

    # Chunk in lockstep across all channels
    windows: list[tuple[int, list[bytes], list[str], int, str]] = []
    for i in range(0, n_samples, sampling_rate):
        ts_ns = _ts_to_ns(ch_timestamps[i])
        payloads = [bytes(ch_arrays[ch][i : i + sampling_rate].tobytes()) for ch in present]
        windows.append((ts_ns, payloads, present, sampling_rate, dtype))

    logger.info(
        'loaded %s: %d windows (%ds), channels=%s, %d samples/channel',
        path,
        len(windows),
        len(windows),
        present,
        n_samples,
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
    ctx.app['restart'] = threading.Event()
    ctx.app['reload'] = threading.Event()
    ctx.app['start_at_ev'] = threading.Event()
    ctx.app['last_start_at'] = None


@app.subscribe(OnChange('scene'), OnChange('node'), OnChange('run'), OnChange('modality'), OnChange('data_dir'))
def on_data_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    logger.info('data config changed: %s %s', msg.op, msg.key)
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
        logger.debug('loading windows from %s', path)
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
    speed: float,
) -> bool:
    """Play all windows at the given speed. Returns True if completed, False if interrupted."""
    wall_start = time.monotonic()
    data_start_ns = windows[0][0]
    topic: str = ctx.cfg.get('topic') or ctx.ns.base
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
        energy = {
            ch: float(np.std(np.frombuffer(p, dtype=np.dtype(dtype)).astype(np.float64)))
            for ch, p in zip(channels, payloads)
        }
        energy_str = ', '.join(f'{ch}={e:.2f}' for ch, e in energy.items())
        ctx.publish(ctx.ns.topic('energy'), {'source': ctx.ns.base, 'timestamp': ts_ns, 'energy': energy})
        logger.debug('%s: t=%.2f (%s)', topic, float(ts_ns / _NS_PER_S), energy_str)
    return True


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


def _wait_for_start_at(
    ctx: AciesContext,
    cancel: list[threading.Event],
) -> bool:
    """Wait until a new start_at is available, then sleep until that wallclock.

    Returns True if ready to play, False if interrupted by stop/restart.
    """
    start_at_ev: threading.Event = ctx.app['start_at_ev']
    wake = [*cancel, start_at_ev]

    # --- wait for start_at_ev (set by on_start_at_change) ---
    while not any(e.is_set() for e in cancel):
        if start_at_ev.is_set():
            start_at_ev.clear()
            start_at: float | None = ctx.cfg.get('start_at')
            if start_at is not None:
                ctx.app['last_start_at'] = start_at
                break
        logger.debug('waiting for start_at...')
        if _wait_for_any(wake, timeout=1.0):
            if any(e.is_set() for e in cancel):
                return False
            # woken by start_at_ev -- loop to consume it
    else:
        return False

    # --- sleep until the start_at wallclock ---
    delay = start_at - time.time()
    if delay > 0:
        logger.info('waiting %.1fs until start_at=%.3f', delay, start_at)
        if _wait_for_any(cancel, timeout=delay):
            return not any(e.is_set() for e in cancel if e is not cancel[0])
    else:
        logger.warning('start_at is %.1fs in the past; starting immediately', -delay)
    return True


@app.thread
def replay(ctx: AciesContext, stop: threading.Event) -> None:
    restart: threading.Event = ctx.app['restart']
    reload_ev: threading.Event = ctx.app['reload']
    cancel = [stop, restart]

    while not stop.is_set():
        restart.clear()

        # --- reload data if requested ---
        if reload_ev.is_set():
            reload_ev.clear()
            if not _try_reload(ctx):
                _ = _wait_for_any(cancel, timeout=3600)
                continue

        windows = ctx.app['windows']
        speed: float = ctx.cfg.get('speed', 1.0)
        loop: bool = ctx.cfg.get('loop', False)
        start_at: float | None = ctx.cfg.get('start_at')

        # --- determine when to start playback ---
        first_run = ctx.app['last_start_at'] is None
        if first_run and start_at is not None:
            # CLI start_at: use once on first run
            ctx.app['last_start_at'] = start_at
            if not _sleep_until(start_at, cancel):
                if stop.is_set():
                    return
                continue
        elif not first_run:
            # After reload or restart: wait for a new start_at from control plane
            logger.info('data loaded, waiting for start_at to begin playback')
            if not _wait_for_start_at(ctx, cancel):
                if stop.is_set():
                    return
                continue

        # --- play ---
        completed = _play_windows(ctx, cancel, windows, speed)
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
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
