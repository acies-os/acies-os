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
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, cast

import click
import msgspec.json
import numpy as np
import numpy.typing as npt
import sounddevice as sd  # pyright: ignore[reportMissingTypeStubs]
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries, setup_logging

from .db import DbRow, flush, open_db

logger = logging.getLogger(__name__)


class PaTimeInfo(Protocol):
    inputBufferAdcTime: float
    outputBufferDacTime: float
    currentTime: float


DB_BATCH = 60  # rows to accumulate before flushing (~60s of data, ~2MB in RAM)
DB_WAL_CHECKPOINT = 16000  # WAL checkpoint threshold in pages (~64MB); reduces I/O spikes on Pi
SAMPLE_DTYPE = 'int16'
BLOCK_SIZE = 1024


# queue of (capture timestamp, audio chunk) tuples from the audio callback to the publisher thread
SAMPLE_QUEUE: queue.Queue[tuple[int, npt.NDArray[np.int16]]] = queue.Queue()

# PA0 and WALL0_NS are initialized at the first callback to calibrate the PortAudio clock to the wall clock.
pa_epoch: float | None = None
wall_epoch_ns: int | None = None


def _audio_callback(indata: npt.NDArray[np.int16], frames: int, cb_time: object, status: sd.CallbackFlags) -> None:
    global pa_epoch, wall_epoch_ns

    if status:
        logger.warning('sounddevice status: %s', status)

    if frames != BLOCK_SIZE:
        logger.warning('unexpected frame count: got %d, expected %d', frames, BLOCK_SIZE)

    block_ns = int(frames * 1_000_000_000 / _sample_rate)
    t = cast(PaTimeInfo, cb_time)
    pa_now = t.inputBufferAdcTime

    if pa_epoch is None:
        # Calibrate PA clock to wall clock at first callback.
        # pa_now is the ADC capture time of this block's first sample;
        # time.time_ns() at this moment is the wall clock equivalent of
        # pa_now + one block duration (callback fires after capture completes).
        # We store the offset so all subsequent timestamps use the same anchor.
        wall_epoch_ns, pa_epoch = time.time_ns() - block_ns, pa_now

    assert wall_epoch_ns is not None and pa_epoch is not None
    capture_ts_ns = wall_epoch_ns + int((pa_now - pa_epoch) * 1_000_000_000)

    # indata shape: (frames, n_channels); mix down to mono
    mono = indata[:, 0].copy()  # first channel only
    # mono = np.rint(indata.astype(np.float32).mean(axis=1)).clip(-32768, 32767).astype(np.int16)

    SAMPLE_QUEUE.put((capture_ts_ns, mono))


@dataclass
class MicState:
    stream: sd.InputStream
    con: sqlite3.Connection
    sample_rate: int
    topic: str
    db_buf: list[DbRow] = field(default_factory=list)
    sample_buf: list[npt.NDArray[np.int16]] = field(default_factory=list)
    buf_frames: int = 0
    buf_start_ts_ns: int = 0


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    device = ctx.app.config['device']
    output = ctx.app.config['output'] or f'/data/{ctx.ns.host}-{ctx.ns.name}.db'
    device_key: str | int | None = None if device == 'default' else device

    try:
        dev_info = sd.query_devices(device_key, 'input')  # pyright: ignore[reportUnknownMemberType]
    except (sd.PortAudioError, ValueError) as e:
        logger.error('audio device %r not found: %s', device, e)
        logger.error('available input devices:')
        for dev in sd.query_devices():  # pyright: ignore[reportUnknownMemberType]
            if dev['max_input_channels'] > 0:
                logger.error('  [%d] %s', dev['index'], dev['name'])
        logger.error('use --device <index or name> to select a device')
        raise SystemExit(1)
    sample_rate = int(dev_info['default_samplerate'])
    n_channels = int(dev_info['max_input_channels'])
    global _sample_rate
    _sample_rate = sample_rate

    stream = sd.InputStream(
        device=device_key,
        channels=n_channels,
        dtype=SAMPLE_DTYPE,
        samplerate=sample_rate,
        blocksize=BLOCK_SIZE,
        callback=_audio_callback,
    )
    stream.start()
    logger.info(
        'mic stream started on device %r: %d Hz, block size %d, %d input channels mixed to mono',
        device,
        sample_rate,
        BLOCK_SIZE,
        n_channels,
    )

    try:
        con = open_db(output, check_same_thread=False, wal_autocheckpoint=DB_WAL_CHECKPOINT)
        logger.info('opened database %r', output)
    except Exception:
        logger.exception('failed to open database %r; shutting down', output)
        stream.stop()
        stream.close()
        raise SystemExit(1)

    ctx.app.data['state'] = MicState(
        stream=stream,
        con=con,
        sample_rate=sample_rate,
        topic=ctx.app.config.get('topic') or ctx.ns.base,
    )


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    state: MicState = ctx.app.data['state']
    state.stream.stop()
    state.stream.close()
    logger.info('mic stream stopped')
    if state.sample_buf:
        logger.debug('discarding %d partial frames at shutdown', state.buf_frames)
        state.sample_buf.clear()
        state.buf_frames = 0
    if state.db_buf:
        n_rows = flush(state.con, state.db_buf)
        logger.debug('flushed %d remaining rows to database', n_rows)
        state.db_buf.clear()
    state.con.close()
    logger.info('database connection closed')


@app.thread
def publish(ctx: AciesContext, stop: threading.Event) -> None:
    state: MicState = ctx.app.data['state']
    while not stop.is_set():
        try:
            ts_ns, chunk = SAMPLE_QUEUE.get(timeout=1.0)
        except queue.Empty:
            continue

        if state.buf_frames == 0:
            state.buf_start_ts_ns = ts_ns
        state.sample_buf.append(chunk)
        state.buf_frames += len(chunk)

        if state.buf_frames < state.sample_rate:
            continue

        # --- assemble 1-second window ---
        window = np.concatenate(state.sample_buf)
        window_ts_ns = state.buf_start_ts_ns
        samples_1s = window[: state.sample_rate]
        leftover = window[state.sample_rate :]
        state.sample_buf = [leftover] if len(leftover) else []
        state.buf_frames = len(leftover)
        if state.buf_frames:
            state.buf_start_ts_ns = window_ts_ns + 1_000_000_000

        topic = state.topic

        # publish to topic
        ctx.publish(
            topic,
            AciesTimeSeries(
                source=ctx.ns.base,
                timestamp=window_ts_ns,
                payload=[samples_1s.tobytes()],
                channels=['mono'],
                sampling_rate=state.sample_rate,
                dtype=SAMPLE_DTYPE,
            ),
        )

        # log latency
        publish_ns = time.time_ns()
        capture_to_publish_ms = (publish_ns - window_ts_ns) / 1_000_000
        ready_to_publish_ms = (publish_ns - window_ts_ns - 1_000_000_000) / 1_000_000
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

        # save to db
        samples = samples_1s.tolist()
        metadata = {'channel': 'mono', 'sampling_rate': state.sample_rate}
        state.db_buf.append(
            (
                topic,
                SAMPLE_DTYPE,
                window_ts_ns,
                ctx.ns.base,
                msgspec.json.encode(samples),
                msgspec.json.encode(metadata),
            )
        )
        if len(state.db_buf) >= DB_BATCH:
            n_rows = flush(state.con, state.db_buf)
            logger.debug('flushed %d rows to database', n_rows)
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
    default=None,
    help='SQLite database output path. Defaults to /data/<acies-host>-<acies-name>.db.',
)
@click.option('--topic', default=None, help='Publish topic. Defaults to <host>/<name>.')
def main(device: str, output: str | None, topic: str | None) -> None:
    app.state.config.update({'device': device, 'output': output, 'topic': topic})
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
