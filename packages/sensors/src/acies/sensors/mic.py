"""Microphone sensor node for AciesOS.

Captures audio from a sounddevice input, accumulates 1-second windows
(channel 0 only), publishes AciesTimeSeries messages on ``<host>/<name>``,
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

import click
import msgspec.json
import numpy as np
import sounddevice as sd  # pyright: ignore[reportMissingTypeStubs]
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries

from .db import DbRow, flush, open_db

logger = logging.getLogger(__name__)

_DB_BATCH = 10  # rows to accumulate before flushing (~10s of data, ~320KB in RAM)
_DB_WAL_CHECKPOINT = 16000  # WAL checkpoint threshold in pages (~64MB); reduces I/O spikes on Pi
_SAMPLE_DTYPE = 'int16'


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
    _con = open_db(output, check_same_thread=False, wal_autocheckpoint=_DB_WAL_CHECKPOINT)
    logger.info('mic stream started on device %r at %d Hz', device, _sample_rate)


def _teardown(ctx: AciesContext) -> None:
    if _stream is not None:
        _stream.stop()
        _stream.close()
    if _db_buf and _con is not None:
        _flush(_con, _db_buf)
        _db_buf.clear()
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
                ctx.ns.base,
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
