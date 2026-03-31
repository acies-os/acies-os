"""VibroFM vehicle classifier node for AciesOS.

Subscribes to AciesTimeSeries messages on configured geo and/or mic topics,
buffers two 1-second windows per modality, and runs a FoundationSense model
every second. Raw inference results are accumulated in a rolling ensemble
buffer; the soft-voted result is published once enough results are available.

Modalities required by the model are determined from the model config file
at startup. Both --geo and --mic topics must be provided; unused topics are
silently ignored.

Usage::

    acies-vfm --weight /path/to/model.pt
              --geo <topic> --mic <topic>
              [--output TOPIC]
              [--labels car,truck,person]
              [--geo-energy-thresh FLOAT] [--mic-energy-thresh FLOAT]
              [--ensemble-win N] [--ensemble-size N]
              [--modality seismic|audio]
              [--freq-mae]
              [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import torch
from acies.buffers import TemporalBuffer
from acies.corev2 import AciesApp, AciesContext, AciesTimeSeries, setup_logging
from acies.corev2.msg import AciesInference, AciesPrediction
from acies.FoundationSense.inference import ModelForInference  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

# samples required per modality for one inference call: 2 x 1-second windows
INPUT_LEN = 2

DEFAULT_LABELS = ['polaris', 'warthog', 'truck', 'husky']

# mapping from FoundationSense dataset modality names to acies modality keys
_MOD_MAPPING: dict[str, str] = {
    'seismic': 'geo',
    'sei': 'geo',
    'acoustic': 'mic',
    'audio': 'mic',
    'aco': 'mic',
}


@dataclass
class VfmState:
    model: ModelForInference
    modalities: list[str]  # 'geo' and/or 'mic', derived from model config
    labels: list[str]
    geo_topic: str
    mic_topic: str
    output_topic: str
    geo_energy_thresh: float
    mic_energy_thresh: float
    # rolling buffer of raw logits for soft-vote ensemble;
    # maxlen = ensemble_win so entries older than the window expire automatically
    ensemble_buf: deque[list[list[float]]]
    ensemble_size: int  # minimum entries required before publishing
    # buffer holds up to INPUT_LEN + 2 seconds per topic to tolerate jitter
    buffer: TemporalBuffer = field(default_factory=lambda: TemporalBuffer(size=INPUT_LEN + 2))


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    weight: str = ctx.app.config['weight']
    freq_mae: bool = ctx.app.config.get('freq_mae', False)
    modality: str | None = ctx.app.config.get('modality')
    geo_topic: str = ctx.app.config['geo_topic']
    mic_topic: str = ctx.app.config['mic_topic']
    output_topic: str = ctx.app.config.get('output_topic') or f'{ctx.ns.base}/vehicle'
    labels: list[str] = ctx.app.config.get('labels') or []
    geo_energy_thresh: float = ctx.app.config.get('geo_energy_thresh', 0.0)
    mic_energy_thresh: float = ctx.app.config.get('mic_energy_thresh', 0.0)
    ensemble_win: int = ctx.app.config.get('ensemble_win', 1)
    ensemble_size: int = ctx.app.config.get('ensemble_size', 1)

    model = ModelForInference(Path(weight), freq_mae, modality=modality)
    raw_mods: list[str] = model.args.dataset_config['modality_names']
    modalities = [_MOD_MAPPING[m] for m in raw_mods]
    logger.info(
        'loaded model from %s; modalities=%s freq_mae=%s #params=%d',
        weight,
        modalities,
        freq_mae,
        sum(p.numel() for p in model.parameters()),
    )

    ctx.app.data['state'] = VfmState(
        model=model,
        modalities=modalities,
        labels=labels,
        geo_topic=geo_topic,
        mic_topic=mic_topic,
        output_topic=output_topic,
        geo_energy_thresh=geo_energy_thresh,
        mic_energy_thresh=mic_energy_thresh,
        ensemble_buf=deque(maxlen=ensemble_win),
        ensemble_size=ensemble_size,
    )
    logger.info(
        'publishing to %s; ensemble_win=%d ensemble_size=%d geo_thresh=%.1f mic_thresh=%.1f',
        output_topic,
        ensemble_win,
        ensemble_size,
        geo_energy_thresh,
        mic_energy_thresh,
    )


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('vibrofm stopped')


@app.subscribe('{geo_topic}')
def on_geo(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    state: VfmState = ctx.app.data['state']
    if 'geo' not in state.modalities:
        return
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    energy = float(np.std(samples))
    if energy < state.geo_energy_thresh:
        logger.debug('geo energy %.1f below threshold %.1f; dropping window', energy, state.geo_energy_thresh)
        return
    ts_s = msg.timestamp // 1_000_000_000
    state.buffer.add(state.geo_topic, ts_s, samples)


@app.subscribe('{mic_topic}')
def on_mic(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    state: VfmState = ctx.app.data['state']
    if 'mic' not in state.modalities:
        return
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    energy = float(np.std(samples))
    if energy < state.mic_energy_thresh:
        logger.debug('mic energy %.1f below threshold %.1f; dropping window', energy, state.mic_energy_thresh)
        return
    ts_s = msg.timestamp // 1_000_000_000
    state.buffer.add(state.mic_topic, ts_s, samples)


@app.schedule(1.0)
def run_inference(ctx: AciesContext) -> None:
    state: VfmState = ctx.app.data['state']
    _mod_to_topic = {'geo': state.geo_topic, 'mic': state.mic_topic}
    keys = [_mod_to_topic[m] for m in state.modalities]

    try:
        samples = state.buffer.pop(keys, INPUT_LEN)
    except ValueError:
        logger.debug('not enough buffered data for inference')
        return

    # --- build FoundationSense model input ---
    # expected format: {'shake': {'seismic': tensor, 'audio': tensor}}
    data: dict[str, dict[str, torch.Tensor]] = {'shake': {}}
    for mod in state.modalities:
        topic = _mod_to_topic[mod]
        arr: npt.NDArray[np.float32] = np.concatenate([v for _, v in sorted(samples[topic].items())]).astype(np.float32)
        if mod == 'geo':
            # 2s x 200 Hz = 400 -> downsample x2 -> 200 -> (1, 1, 10, 20)
            seismic_np = arr[::2].reshape(1, 1, 10, 20)
            data['shake']['seismic'] = torch.from_numpy(seismic_np)  # pyright: ignore[reportUnknownMemberType]
        else:
            # 2s x 16000 Hz = 32000 -> downsample x2 -> 16000 -> (1, 1, 10, 1600)
            acoustic_np = arr[::2].reshape(1, 1, 10, 1600)
            data['shake']['audio'] = torch.from_numpy(acoustic_np)  # pyright: ignore[reportUnknownMemberType]

    t0 = time.perf_counter_ns()
    logit = state.model(data)  # returns [[score_0, score_1, ...]]
    infer_ms = (time.perf_counter_ns() - t0) / 1_000_000

    logits: list[list[float]] = [[float(x) for x in logit[0]]]
    state.ensemble_buf.append(logits)
    logger.debug('inference: logits=%s infer_ms=%.1f ensemble=%d', logits, infer_ms, len(state.ensemble_buf))

    if len(state.ensemble_buf) < state.ensemble_size:
        return

    # --- soft-vote ensemble: average logits across the window ---
    # shape: (ensemble_win, num_targets, num_classes) -> mean over axis 0
    ensemble_logits: list[list[float]] = np.array(list(state.ensemble_buf)).mean(axis=0).tolist()

    raw = np.array(ensemble_logits[0])
    probs = np.exp(raw - raw.max())
    probs /= probs.sum()
    preds = [AciesPrediction(label=label, score=float(score)) for label, score in zip(state.labels, probs)]
    ctx.publish(
        state.output_topic,
        AciesInference(source=ctx.ns.base, timestamp=ctx.now(), predictions=preds),
    )


@app.cli()
@click.option('--weight', required=True, type=click.Path(exists=True), help='Model weight file path.')
@click.option('--geo', 'geo_topic', required=True, help='Geo input topic (AciesTimeSeries).')
@click.option('--mic', 'mic_topic', required=True, help='Mic input topic (AciesTimeSeries).')
@click.option(
    '--output',
    'output_topic',
    default=None,
    help='Output topic for AciesInference results. Defaults to <host>/<name>/vehicle.',
)
@click.option(
    '--labels',
    default=','.join(DEFAULT_LABELS),
    show_default=True,
    help='Comma-separated class names matching model output order.',
)
@click.option(
    '--geo-energy-thresh',
    default=0.0,
    type=float,
    show_default=True,
    help='Geo std threshold; windows below are dropped.',
)
@click.option(
    '--mic-energy-thresh',
    default=0.0,
    type=float,
    show_default=True,
    help='Mic std threshold; windows below are dropped.',
)
@click.option(
    '--ensemble-win',
    default=1,
    type=int,
    show_default=True,
    help='Rolling window size in seconds for soft-vote ensemble.',
)
@click.option(
    '--ensemble-size', default=1, type=int, show_default=True, help='Minimum results in window before publishing.'
)
@click.option(
    '--modality',
    default=None,
    type=str,
    help='Single modality override for the model (e.g. seismic, audio). Omit for multimodal.',
)
@click.option(
    '--freq-mae',
    is_flag=True,
    default=False,
    show_default=True,
    help='Use frequency-domain MAE model variant.',
)
def main(
    weight: str,
    geo_topic: str,
    mic_topic: str,
    output_topic: str | None,
    labels: str,
    geo_energy_thresh: float,
    mic_energy_thresh: float,
    ensemble_win: int,
    ensemble_size: int,
    modality: str | None,
    freq_mae: bool,
) -> None:
    app.state.config.update(
        {
            'weight': weight,
            'geo_topic': geo_topic,
            'mic_topic': mic_topic,
            'output_topic': output_topic,
            'labels': labels.split(','),
            'geo_energy_thresh': geo_energy_thresh,
            'mic_energy_thresh': mic_energy_thresh,
            'ensemble_win': ensemble_win,
            'ensemble_size': ensemble_size,
            'modality': modality,
            'freq_mae': freq_mae,
        }
    )
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
