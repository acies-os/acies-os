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
"""

from __future__ import annotations

import logging
import socket
import sqlite3
from dataclasses import dataclass, field

import click
import msgspec.json
import numpy as np
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries
from rawshake.geophone import Channel, GeoReader, get_samples

from .db import DbRow, flush, open_db

logger = logging.getLogger(__name__)

GEO_CHANNELS: tuple[Channel, Channel] = ('SH3', 'EH3')  # RS1D: SH3, RS4D: EH3; order is preference
SAMPLING_RATE = 200  # Hz, fixed for all RaspberryShake devices
SAMPLE_DTYPE = 'int32'
DB_BATCH = 60  # rows to accumulate before flushing (~60s of data)
DB_WAL_CHECKPOINT = 16000  # WAL checkpoint threshold in pages (~64MB); reduces I/O spikes on Pi


@dataclass
class ReaderState:
    reader: GeoReader
    con: sqlite3.Connection
    topic: str
    db_buf: list[DbRow] = field(default_factory=list)


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    port: str = ctx.app.config['port']
    baud: int = ctx.app.config['baud']
    output: str = ctx.app.config['output']
    reader = GeoReader(port=port, baudrate=baud)
    reader.start()
    logger.info('geo reader started on %s @ %d baud', port, baud)

    ctx.app.data['state'] = ReaderState(
        reader=reader,
        con=open_db(output, check_same_thread=False, wal_autocheckpoint=DB_WAL_CHECKPOINT),
        topic=ctx.app.config.get('topic') or ctx.ns.base,
    )


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    state: ReaderState = ctx.app.data['state']
    state.reader.stop()
    logger.info('geo reader stopped')
    if state.db_buf:
        _ = flush(state.con, state.db_buf)
        state.db_buf.clear()
    state.con.close()
    logger.info('database connection closed')


@app.schedule(interval=0.5)
def publish(ctx: AciesContext) -> None:
    state: ReaderState = ctx.app.data['state']
    while (msg := state.reader.get(timeout=0)) is not None:
        ts_ns, channel_samples = get_samples(msg)
        # Pick the first preferred channel present in the message.
        # GEO_CHANNELS order encodes preference: SH3 (RS1D) before EH3 (RS4D).
        channel: Channel | None = next((c for c in GEO_CHANNELS if c in channel_samples), None)
        if channel is None:
            logger.warning('no geo channel in message; available: %s', list(channel_samples))
            continue

        samples = channel_samples[channel]
        topic = state.topic

        ctx.publish(
            topic,
            AciesTimeSeries(
                source=ctx.ns.base,
                timestamp=ts_ns,
                payload=[np.array(samples, dtype=SAMPLE_DTYPE).tobytes()],
                channels=[channel],
                sampling_rate=SAMPLING_RATE,
                dtype=SAMPLE_DTYPE,
            ),
        )

        metadata = {'channel': channel, 'sampling_rate': SAMPLING_RATE}
        state.db_buf.append(
            (
                topic,
                SAMPLE_DTYPE,
                ts_ns,
                ctx.ns.base,
                msgspec.json.encode(samples),
                msgspec.json.encode(metadata),
            )
        )
        if len(state.db_buf) >= DB_BATCH:
            _ = flush(state.con, state.db_buf)
            state.db_buf.clear()


@app.cli()
@click.option('--port', default='/dev/serial0', show_default=True, help='Serial port path.')
@click.option('--baud', default=230400, type=int, show_default=True, help='Baud rate.')
@click.option(
    '--output',
    default=f'/data/{socket.gethostname().removesuffix(".local")}-geo.db',
    show_default=True,
    help='SQLite database output path.',
)
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
def main(port: str, baud: int, output: str, topic: str | None) -> None:
    app.state.config.update({'port': port, 'baud': baud, 'output': output, 'topic': topic})
    app.run()


if __name__ == '__main__':
    main()
