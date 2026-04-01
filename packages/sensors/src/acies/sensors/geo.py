"""Geophone sensor node for AciesOS.

Reads 1-second windows from a Raspberry Shake (RS1D or RS4D) over serial,
publishes AciesTimeSeries messages on ``<host>/<name>``, and writes to SQLite.

Device channel mapping:
  RS1D -> SH3 (single geophone)
  RS4D -> EH3 (geophone; EN1/EN2/EN3 are MEMs, ignored)

Sampling rate is fixed at 200 Hz for all RaspberryShake devices.

Usage::

    acies-geo [--port /dev/serial0] [--baud 230400] [--output /data/host-geo.db]
              [--acies-host HOST] [--acies-name NAME]
              [--condition] [--hp HZ] [--lp HZ] [--condition-seconds N]
"""

from __future__ import annotations

import logging
import threading
import time

import click
import msgspec.json
import numpy as np
import numpy.typing as npt
from acies.corev2 import AciesApp, AciesContext, setup_logging
from acies.corev2.msg import AciesTimeSeries
from rawshake.geophone import Channel, GeoReader, get_samples
from rawshake.processing import RollingConditioner

from .db import DbRow, flush, open_db

logger = logging.getLogger(__name__)

GEO_CHANNELS: tuple[Channel, Channel] = ('SH3', 'EH3')  # RS1D: SH3, RS4D: EH3; order is preference
SAMPLING_RATE = 200  # Hz, fixed for all RaspberryShake devices
SAMPLE_DTYPE_RAW = 'int32'
SAMPLE_DTYPE_CONDITIONED = 'float64'
DB_BATCH = 60  # rows to accumulate before flushing (~60s of data)
DB_WAL_CHECKPOINT = 16000  # WAL checkpoint threshold in pages (~64MB); reduces I/O spikes on Pi


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    port: str = ctx.cfg['port']
    baud: int = ctx.cfg['baud']
    output: str = ctx.cfg.get('output') or f'/data/{ctx.ns.host}-{ctx.ns.name}.db'
    reader = GeoReader(port=port, baudrate=baud)
    reader.start()
    logger.info('geo reader started on %s @ %d baud', port, baud)

    try:
        con = open_db(output, check_same_thread=False, wal_autocheckpoint=DB_WAL_CHECKPOINT)
        logger.info('opened database %r', output)
    except Exception:
        logger.exception('failed to open database %r; shutting down', output)
        reader.stop()
        raise SystemExit(1)

    conditioner: RollingConditioner | None = None
    if ctx.cfg.get('condition'):
        hp: float | None = ctx.cfg.get('hp')
        lp: float | None = ctx.cfg.get('lp')
        seconds: int = ctx.cfg.get('condition_seconds') or 5
        conditioner = RollingConditioner(fs=SAMPLING_RATE, seconds=seconds, hp=hp, lp=lp)
        logger.info(
            'RollingConditioner enabled: hp=%s Hz, lp=%s Hz, buffer=%ds',
            hp,
            lp,
            seconds,
        )

    ctx.app['reader'] = reader
    ctx.app['con'] = con
    ctx.app['topic'] = ctx.cfg.get('topic') or ctx.ns.base
    ctx.app['conditioner'] = conditioner
    ctx.app['db_buf'] = []


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    ctx.app['reader'].stop()
    logger.info('geo reader stopped')
    db_buf: list[DbRow] = ctx.app['db_buf']
    if db_buf:
        try:
            n_rows = flush(ctx.app['con'], db_buf)
            logger.debug('flushed %d remaining rows to database', n_rows)
        except Exception:
            logger.exception('database flush failed during shutdown; %d rows lost', len(db_buf))
        db_buf.clear()
    ctx.app['con'].close()
    logger.info('database connection closed')


@app.thread
def publish(ctx: AciesContext, stop: threading.Event) -> None:
    db_buf: list[DbRow] = ctx.app['db_buf']
    while not stop.is_set():
        msg = ctx.app['reader'].get(timeout=1.0)
        if msg is None:
            continue

        ts_ns, channel_samples = get_samples(msg)
        # Pick the first preferred channel present in the message.
        # GEO_CHANNELS order encodes preference: SH3 (RS1D) before EH3 (RS4D).
        channel: Channel | None = next((c for c in GEO_CHANNELS if c in channel_samples), None)
        if channel is None:
            logger.warning('no geo channel in message; available: %s', list(channel_samples))
            continue

        conditioner: RollingConditioner | None = ctx.app['conditioner']
        if conditioner is not None:
            # push all channels so every buffer stays current; select after
            conditioned = conditioner.push(channel_samples)
            samples_array: npt.NDArray[np.float64] = conditioned[channel]
            dtype = SAMPLE_DTYPE_CONDITIONED
        else:
            samples_array = np.array(channel_samples[channel], dtype=SAMPLE_DTYPE_RAW)
            dtype = SAMPLE_DTYPE_RAW

        topic: str = ctx.app['topic']
        ctx.publish(
            topic,
            AciesTimeSeries(
                source=ctx.ns.base,
                timestamp=ts_ns,
                payload=[samples_array.tobytes()],
                channels=[channel],
                sampling_rate=SAMPLING_RATE,
                dtype=dtype,
            ),
        )

        # log latency
        publish_ns = time.time_ns()
        capture_to_publish_ms = (publish_ns - ts_ns) / 1_000_000
        ready_to_publish_ms = (publish_ns - ts_ns - 1_000_000_000) / 1_000_000
        if ready_to_publish_ms > 1000:
            logger.warning(
                'window latency: capture_to_publish=%.0f ms ready_to_publish=%.0f ms',
                capture_to_publish_ms,
                ready_to_publish_ms,
            )
        else:
            logger.debug(
                'window latency: capture_to_publish=%.0f ms ready_to_publish=%.0f ms',
                capture_to_publish_ms,
                ready_to_publish_ms,
            )

        metadata = {'channel': channel, 'sampling_rate': SAMPLING_RATE}
        db_buf.append(
            (
                topic,
                dtype,
                ts_ns,
                ctx.ns.base,
                msgspec.json.encode(samples_array.tolist()),
                msgspec.json.encode(metadata),
            )
        )
        if len(db_buf) >= DB_BATCH:
            try:
                n_rows = flush(ctx.app['con'], db_buf)
                logger.debug('flushed %d rows to database', n_rows)
                db_buf.clear()
            except Exception:
                logger.exception('database flush failed; shutting down')
                app.stop()
                return


@app.cli()
@click.option('--port', default='/dev/serial0', show_default=True, help='Serial port path.')
@click.option('--baud', default=230400, type=int, show_default=True, help='Baud rate.')
@click.option(
    '--output',
    default=None,
    help='SQLite database output path. Defaults to /data/<acies-host>-<acies-name>.db.',
)
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
@click.option(
    '--condition/--no-condition',
    default=False,
    show_default=True,
    help='Apply RollingConditioner (DC removal, detrend, optional bandpass).',
)
@click.option('--hp', default=None, type=float, help='High-pass corner frequency in Hz.')
@click.option('--lp', default=None, type=float, help='Low-pass corner frequency in Hz.')
@click.option(
    '--condition-seconds',
    default=5,
    type=int,
    show_default=True,
    help='Rolling buffer length in seconds for the conditioner.',
)
def main(
    port: str,
    baud: int,
    output: str | None,
    topic: str | None,
    condition: bool,
    hp: float | None,
    lp: float | None,
    condition_seconds: int,
) -> None:
    app.state.config.update(
        {
            'port': port,
            'baud': baud,
            'output': output,
            'topic': topic,
            'condition': condition,
            'hp': hp,
            'lp': lp,
            'condition_seconds': condition_seconds,
        }
    )
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
