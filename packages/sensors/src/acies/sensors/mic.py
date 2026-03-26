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
from dataclasses import dataclass, field

import click
import msgspec.json
import numpy as np
import numpy.typing as npt
import sounddevice as sd  # pyright: ignore[reportMissingTypeStubs]
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries

from .db import DbRow, flush, open_db

logger = logging.getLogger(__name__)

DB_BATCH = 60  # rows to accumulate before flushing (~60s of data, ~2MB in RAM)
DB_WAL_CHECKPOINT = 16000  # WAL checkpoint threshold in pages (~64MB); reduces I/O spikes on Pi
SAMPLE_DTYPE = 'int16'


_sample_queue: queue.Queue[npt.NDArray[np.int16]] = queue.Queue()


def _audio_callback(indata: npt.NDArray[np.int16], _frames: int, _time: object, status: sd.CallbackFlags) -> None:
    if status:
        logger.warning('sounddevice status: %s', status)
    # indata shape: (frames, 1) since channels=1; take channel 0 as a copy
    _sample_queue.put(indata[:, 0].copy())


@dataclass
class MicState:
    stream: sd.InputStream
    con: sqlite3.Connection
    sample_rate: int
    topic: str
    sample_buf: list[int] = field(default_factory=list)
    db_buf: list[DbRow] = field(default_factory=list)


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    device = ctx.app.config['device']
    output = ctx.app.config['output']
    device_key: str | int | None = None if device == 'default' else device

    dev_info = sd.query_devices(device_key, 'input')  # pyright: ignore[reportUnknownMemberType]
    sample_rate = int(dev_info['default_samplerate'])

    stream = sd.InputStream(
        device=device_key,
        channels=1,
        dtype=SAMPLE_DTYPE,
        samplerate=sample_rate,
        blocksize=sample_rate,  # 1 second per callback
        callback=_audio_callback,
    )
    stream.start()
    logger.info('mic stream started on device %r at %d Hz', device, sample_rate)

    ctx.app.data['state'] = MicState(
        stream=stream,
        con=open_db(output, check_same_thread=False, wal_autocheckpoint=DB_WAL_CHECKPOINT),
        sample_rate=sample_rate,
        topic=ctx.app.config.get('topic') or ctx.ns.base,
    )


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    state: MicState = ctx.app.data['state']
    state.stream.stop()
    state.stream.close()
    if state.db_buf:
        _ = flush(state.con, state.db_buf)
        state.db_buf.clear()
    state.con.close()


@app.schedule(interval=0.5)
def publish(ctx: AciesContext) -> None:
    state: MicState = ctx.app.data['state']
    sr = state.sample_rate

    while True:
        try:
            state.sample_buf.extend(_sample_queue.get_nowait().tolist())
        except queue.Empty:
            break

    while len(state.sample_buf) >= sr:
        samples = state.sample_buf[:sr]
        del state.sample_buf[:sr]

        topic = state.topic
        ts_ns = ctx.now()

        ctx.publish(
            topic,
            AciesTimeSeries(
                source=ctx.ns.base,
                timestamp=ts_ns,
                payload=[np.array(samples, dtype=SAMPLE_DTYPE).tobytes()],
                channels=[0],
                sampling_rate=sr,
                dtype=SAMPLE_DTYPE,
            ),
        )

        metadata = {'channel': 0, 'sampling_rate': sr}
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
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
def main(device: str, output: str, topic: str | None) -> None:
    app.state.config.update({'device': device, 'output': output, 'topic': topic})
    app.run()


if __name__ == '__main__':
    main()
