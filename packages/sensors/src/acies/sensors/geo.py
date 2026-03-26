"""Geophone sensor node for AciesOS.

Reads 1-second windows from a Raspberry Shake (RS1D or RS4D) over serial,
publishes AciesTensor messages on ``<host>/<name>``, and writes to SQLite.

Device channel mapping:
  RS1D -> SH3 (single geophone)
  RS4D -> EH3 (geophone; EN1/EN2/EN3 are MEMs, ignored)

Sampling rate is fixed at 200 Hz for all RaspberryShake devices.

Usage::

    acies-geo [--port /dev/serial0] [--baud 230400] [--output /data/host-geo.db]
              [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import threading
from pathlib import Path

import click
from acies.corev2 import AciesApp, AciesContext, AciesTensor
from rawshake.geophone import GeoReader, get_samples

logger = logging.getLogger(__name__)

GEO_CHANNELS = ('SH3', 'EH3')  # RS1D: SH3, RS4D: EH3; order is preference
SAMPLING_RATE = 200  # Hz, fixed for all RaspberryShake devices
DB_BATCH = 5  # rows to accumulate before flushing to SQLite


# --- SQLite helpers ---


def open_db(path: str) -> sqlite3.Connection:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), check_same_thread=False)
    _ = con.execute(
        """
        CREATE TABLE IF NOT EXISTS message (
            id        INTEGER PRIMARY KEY,
            topic     TEXT NOT NULL,
            msg_type  TEXT NOT NULL,
            timestamp INT  NOT NULL,
            ctl_topic TEXT NOT NULL,
            payload   TEXT,
            metadata  TEXT
        )
        """
    )
    con.commit()
    return con


# Note: payload and metadata are stored as JSON strings
# topic: str, msg_type: str, timestamp: int, ctl_topic: str, payload: str | None, metadata: str | None
DbRow = tuple[str, str, int, str, str | None, str | None]


def flush(con: sqlite3.Connection, rows: list[DbRow]) -> None:
    _ = con.executemany(
        'INSERT INTO message (topic, msg_type, timestamp, ctl_topic, payload, metadata) VALUES (?,?,?,?,?,?)',
        rows,
    )
    con.commit()


# --- module-level sensor state ---

_lock = threading.Lock()


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    port: str = ctx.app.config['port']
    baud: int = ctx.app.config['baud']
    output: str = ctx.app.config['output']
    reader = GeoReader(port=port, baudrate=baud)
    reader.start()
    logger.info('geo reader started on %s @ %d baud', port, baud)

    ctx.app.data['db_buf'] = []
    ctx.app.data['reader'] = reader
    ctx.app.data['con'] = open_db(output)


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    ctx.app.data['reader'].stop()
    ctx.app.data['con'].close()


@app.schedule(interval=0.5)
def publish(ctx: AciesContext) -> None:
    reader = ctx.app.data['reader']
    con = ctx.app.data['con']
    with _lock:
        while (msg := reader.get(timeout=0)) is not None:
            ts_ns, by_channel = get_samples(msg)
            channel = next((c for c in GEO_CHANNELS if c in by_channel), None)
            if channel is None:
                logger.warning('no geo channel in message; available: %s', list(by_channel))
                continue

            samples = by_channel[channel]
            metadata = {'channel': channel, 'sampling_rate': SAMPLING_RATE}
            topic = ctx.ns.base

            ctx.publish(
                topic,
                AciesTensor(
                    source=ctx.ns.base,
                    timestamp=ts_ns,
                    payload=samples,
                    metadata=metadata,
                ),
            )

            db_buf = ctx.app.data['db_buf']

            db_buf.append(
                (
                    topic,
                    'i32',
                    ts_ns,
                    ctx.ns.ctl.base,
                    json.dumps(samples),
                    json.dumps(metadata),
                )
            )
            if len(db_buf) >= DB_BATCH and con is not None:
                flush(con, db_buf)
                db_buf.clear()
            ctx.app.data['db_buf'] = db_buf


@app.cli()
@click.option('--port', default='/dev/serial0', show_default=True, help='Serial port path.')
@click.option('--baud', default=230400, type=int, show_default=True, help='Baud rate.')
@click.option(
    '--output',
    default=f'/data/{socket.gethostname().removesuffix(".local")}-geo.db',
    show_default=True,
    help='SQLite database output path.',
)
def main(port: str, baud: int, output: str) -> None:
    app.state.config.update({'port': port, 'baud': baud, 'output': output})
    app.run()


if __name__ == '__main__':
    main()
