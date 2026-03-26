"""Microphone sensor node for AciesOS.

Captures audio from a sounddevice input, accumulates 1-second windows
(channel 0 only), publishes AciesTensor messages on ``<host>/<name>``,
and writes to SQLite.

Usage::

    acies-mic [--device default] [--output /data/host-mic.db]
              [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import queue
import socket
import sqlite3
from pathlib import Path

import click
import msgspec.json
import numpy as np
import sounddevice as sd
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries

logger = logging.getLogger(__name__)

_DB_BATCH = 5  # rows to accumulate before flushing to SQLite
_SAMPLE_DTYPE = 'int16'

# --- SQLite helpers ---


def open_db(path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite message database at *path*.

    payload and metadata are stored as BLOB (raw UTF-8 JSON bytes). Use
    CAST(... AS TEXT) to read them as strings from the CLI::

        sqlite3 /data/host-mic.db \\
          "SELECT topic, msg_type, datetime(timestamp/1e9, 'unixepoch'),
                  CAST(payload AS TEXT), CAST(metadata AS TEXT)
           FROM message
           WHERE timestamp BETWEEN <start_ns> AND <end_ns>
           ORDER BY timestamp"
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: connection is opened on the startup hook (main
    # thread) but written from the worker thread running _publish(). Safe because
    # _publish() is the only writer and the scheduler never calls it concurrently.
    con = sqlite3.connect(str(p), check_same_thread=False)
    _ = con.execute(
        """
        CREATE TABLE IF NOT EXISTS message (
            id        INTEGER PRIMARY KEY,
            topic     TEXT NOT NULL,
            msg_type  TEXT NOT NULL,
            timestamp INT  NOT NULL,
            ctl_topic TEXT NOT NULL,
            payload   BLOB,
            metadata  BLOB
        )
        """
    )
    _ = con.execute('CREATE INDEX IF NOT EXISTS idx_message_timestamp ON message (timestamp)')
    con.commit()
    return con


# topic, msg_type, timestamp, ctl_topic, payload, metadata
DbRow = tuple[str, str, int, str, bytes, bytes]


def flush(con: sqlite3.Connection, rows: list[DbRow]) -> None:
    _ = con.executemany(
        'INSERT INTO message (topic, msg_type, timestamp, ctl_topic, payload, metadata) VALUES (?,?,?,?,?,?)',
        rows,
    )
    con.commit()


# --- audio callback (runs on sounddevice thread) ---


_sample_queue: queue.Queue[np.ndarray] = queue.Queue()


def _audio_callback(indata: np.ndarray, frames: int, time: object, status: sd.CallbackFlags) -> None:
    if status:
        logger.warning('sounddevice status: %s', status)
    # indata shape: (frames, 1) since channels=1; take channel 0 as a copy
    _sample_queue.put(indata[:, 0].copy())


# --- module-level sensor state ---

_stream: sd.InputStream | None = None
_con: sqlite3.Connection | None = None
_sample_rate: int = 0
_accumulator: list[int] = []
_db_buf: list[DbRow] = []

# --- handlers ---


def _setup(ctx: AciesContext) -> None:
    global _stream, _con, _sample_rate
    device = ctx.app.config['device']
    output = ctx.app.config['output']
    device_key: str | int | None = None if device == 'default' else device

    dev_info = sd.query_devices(device_key, 'input')
    _sample_rate = int(dev_info['default_samplerate'])  # type: ignore[index]

    _stream = sd.InputStream(
        device=device_key,
        channels=1,
        dtype=_SAMPLE_DTYPE,
        samplerate=_sample_rate,
        blocksize=_sample_rate,  # 1 second per callback
        callback=_audio_callback,
    )
    _stream.start()
    _con = open_db(output)
    logger.info('mic stream started on device %r at %d Hz', device, _sample_rate)


def _teardown(ctx: AciesContext) -> None:
    if _stream is not None:
        _stream.stop()
        _stream.close()
    if _con is not None:
        _con.close()


def _publish(ctx: AciesContext) -> None:
    if _sample_rate == 0:
        return
    sr = _sample_rate
    while True:
        try:
            _accumulator.extend(_sample_queue.get_nowait().tolist())
        except queue.Empty:
            break

    while len(_accumulator) >= sr:
        samples = _accumulator[:sr]
        del _accumulator[:sr]

        topic = ctx.ns.base
        ts_ns = ctx.now()

        ctx.publish(
            topic,
            AciesTimeSeries(
                source=ctx.ns.base,
                timestamp=ts_ns,
                payload=[np.array(samples, dtype=_SAMPLE_DTYPE).tobytes()],
                channels=[0],
                sampling_rate=sr,
                dtype=_SAMPLE_DTYPE,
            ),
        )

        metadata = {'channel': 0, 'sampling_rate': sr}
        _db_buf.append(
            (
                topic,
                _SAMPLE_DTYPE,
                ts_ns,
                ctx.ns.ctl.base,
                msgspec.json.encode(samples),
                msgspec.json.encode(metadata),
            )
        )
        if len(_db_buf) >= _DB_BATCH and _con is not None:
            flush(_con, _db_buf)
            _db_buf.clear()


# --- entry point ---

app = AciesApp()
app.on_startup(_setup)
app.on_shutdown(_teardown)
app.schedule(interval=0.5)(_publish)


@app.cli()
@click.option(
    '--device',
    default='default',
    show_default=True,
    help='Input device name or index. Use "default" for the system default.',
)
@click.option(
    '--output',
    default=f'/data/{socket.gethostname().removesuffix(".local")}-mic.db',
    show_default=True,
    help='SQLite database output path.',
)
def main(device: str, output: str) -> None:
    app.state.config.update({'device': device, 'output': output})
    app.run()


if __name__ == '__main__':
    main()
