"""Noise detector node for AciesOS.

Subscribes to AciesTimeSeries messages on configured geo and/or mic topics,
computes the standard deviation (energy proxy) of each 1-second window, and
publishes a rolling mean energy per modality every second.

Usage::

    acies-noise-detector --geo <topic> --mic <topic>
                         [--win-size N]
                         [--output TOPIC]
                         [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import click
import msgspec
import numpy as np
import numpy.typing as npt
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries, setup_logging

logger = logging.getLogger(__name__)


# --- output message type ---


class AciesNoiseLevel(msgspec.Struct, frozen=True):
    """Rolling mean energy (std of raw samples) per modality.

    ``geo`` and ``mic`` are the mean std over the last ``win_size`` seconds.
    A value of 0.0 means no samples were received in the window.
    """

    source: str
    timestamp: int  # ns since Unix epoch (time.time_ns())
    geo: float
    mic: float


# --- per-run state ---


@dataclass
class NoiseState:
    geo_topic: str
    mic_topic: str
    output_topic: str
    geo_buf: deque[float]  # one std value per received geo window
    mic_buf: deque[float]  # one std value per received mic window


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    geo_topic: str = ctx.app.config['geo_topic']
    mic_topic: str = ctx.app.config['mic_topic']
    output_topic: str = ctx.app.config.get('output_topic') or f'{ctx.ns.base}/noise'
    win_size: int = ctx.app.config.get('win_size', 10)

    ctx.app.data['state'] = NoiseState(
        geo_topic=geo_topic,
        mic_topic=mic_topic,
        output_topic=output_topic,
        geo_buf=deque(maxlen=win_size),
        mic_buf=deque(maxlen=win_size),
    )
    logger.info('publishing noise levels to %s; win_size=%d', output_topic, win_size)


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    logger.info('noise detector stopped')


@app.subscribe('{geo_topic}')
def on_geo(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    state: NoiseState = ctx.app.data['state']
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    state.geo_buf.append(float(np.std(samples)))


@app.subscribe('{mic_topic}')
def on_mic(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    state: NoiseState = ctx.app.data['state']
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    state.mic_buf.append(float(np.std(samples)))


@app.schedule(1.0)
def detect(ctx: AciesContext) -> None:
    state: NoiseState = ctx.app.data['state']
    geo_mean = float(np.mean(state.geo_buf)) if state.geo_buf else 0.0
    mic_mean = float(np.mean(state.mic_buf)) if state.mic_buf else 0.0
    logger.debug('noise: geo=%.1f mic=%.1f', geo_mean, mic_mean)
    ctx.publish(
        state.output_topic,
        AciesNoiseLevel(
            source=ctx.ns.base,
            timestamp=ctx.now(),
            geo=geo_mean,
            mic=mic_mean,
        ),
    )


@app.cli()
@click.option('--geo', 'geo_topic', required=True, help='Geo input topic (AciesTimeSeries).')
@click.option('--mic', 'mic_topic', required=True, help='Mic input topic (AciesTimeSeries).')
@click.option(
    '--output',
    'output_topic',
    default=None,
    help='Output topic for AciesNoiseLevel results. Defaults to <host>/<name>/noise.',
)
@click.option(
    '--win-size',
    default=10,
    type=int,
    show_default=True,
    help='Rolling window size in seconds for mean energy computation.',
)
def main(
    geo_topic: str,
    mic_topic: str,
    output_topic: str | None,
    win_size: int,
) -> None:
    app.state.config.update(
        {
            'geo_topic': geo_topic,
            'mic_topic': mic_topic,
            'output_topic': output_topic,
            'win_size': win_size,
        }
    )
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
