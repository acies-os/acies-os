"""Microphone sensor node for AciesOS.

Captures audio from a sounddevice input, accumulates 1-second windows
(channel 0 only), publishes AciesTensor messages on ``<host>/<name>``,
and writes to SQLite.

Usage::

    acies-mic [--device default] [--output /data/host-mic.db]
              [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import sqlite3
import threading
from pathlib import Path

import click
import numpy as np
import sounddevice as sd
from acies.corev2 import AciesApp, AciesContext, AciesTensor

logger = logging.getLogger(__name__)

_DB_BATCH = 5  # rows to accumulate before flushing to SQLite

# --- SQLite helpers ---


def _open_db(path: str) -> sqlite3.Connection:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), check_same_thread=False)
    con.execute(
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


def _flush(con: sqlite3.Connection, rows: list[tuple]) -> None:
    con.executemany(
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
_db_buf: list[tuple] = []
_lock = threading.Lock()

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
        dtype='int16',
        samplerate=_sample_rate,
        blocksize=_sample_rate,  # 1 second per callback
        callback=_audio_callback,
    )
    _stream.start()
    _con = _open_db(output)
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
    with _lock:
        while True:
            try:
                _accumulator.extend(_sample_queue.get_nowait().tolist())
            except queue.Empty:
                break

        while len(_accumulator) >= sr:
            samples = _accumulator[:sr]
            del _accumulator[:sr]

            metadata = {'channel': 0, 'sampling_rate': sr}
            topic = ctx.ns.base
            ts_ns = ctx.now()

            ctx.publish(
                topic,
                AciesTensor(
                    source=ctx.ns.base,
                    timestamp=ts_ns,
                    payload=samples,
                    metadata=metadata,
                ),
            )

            _db_buf.append(
                (
                    topic,
                    'i16',
                    ts_ns,
                    ctx.ns.ctl.base,
                    json.dumps(samples),
                    json.dumps(metadata),
                )
            )
            if len(_db_buf) >= _DB_BATCH and _con is not None:
                _flush(_con, _db_buf)
                _db_buf.clear()


# --- entry point ---


@click.command(name='acies-mic')
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
@click.option(
    '--acies-host', default=socket.gethostname().removesuffix('.local'), show_default=True, help='Node hostname.'
)
@click.option('--acies-name', default='mic', show_default=True, help='Node name.')
def main(device: str, output: str, acies_host: str, acies_name: str) -> None:
    app = AciesApp(name=acies_name, host=acies_host)
    app.state.config.update({'device': device, 'output': output})
    app.on_startup(_setup)
    app.on_shutdown(_teardown)
    app.schedule(interval=0.5)(_publish)
    app.run()


if __name__ == '__main__':
    main()
