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
import sys
import time
from collections import deque
from pathlib import Path

import click
import numpy as np
import numpy.typing as npt
import torch
from acies.buffers import TemporalBuffer
from acies.corev2 import AciesApp, AciesContext, setup_logging
from acies.corev2.msg import AciesInference, AciesPrediction, AciesTimeSeries
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


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    freq_mae: bool = ctx.cfg.get('freq_mae', False)
    ensemble_win: int = ctx.cfg.get('ensemble_win', 1)

    # ModelForInference calls argparse.parse_args() internally at construction;
    # clear sys.argv so it does not see our Click arguments and error out.
    _saved_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        model = ModelForInference(Path(ctx.cfg['weight']), freq_mae, modality=ctx.cfg.get('modality'))
    finally:
        sys.argv = _saved_argv
    raw_mods: list[str] = model.args.dataset_config['modality_names']
    modalities = [_MOD_MAPPING[m] for m in raw_mods]
    logger.info(
        'loaded model from %s; modalities=%s freq_mae=%s #params=%d',
        ctx.cfg['weight'],
        modalities,
        freq_mae,
        sum(p.numel() for p in model.parameters()),
    )

    output_topic: str = ctx.cfg.get('output_topic') or ctx.ns.topic('vehicle')
    ctx.app['model'] = model
    ctx.app['modalities'] = modalities
    ctx.app['output_topic'] = output_topic
    ctx.app['ensemble_buf'] = deque(maxlen=ensemble_win)
    ctx.app['buffer'] = TemporalBuffer(size=INPUT_LEN + 2)

    logger.info(
        'publishing to %s; ensemble_win=%d ensemble_size=%d geo_thresh=%.1f mic_thresh=%.1f',
        output_topic,
        ensemble_win,
        ctx.cfg.get('ensemble_size', 1),
        ctx.cfg.get('geo_energy_thresh', 0.0),
        ctx.cfg.get('mic_energy_thresh', 0.0),
    )


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('vibrofm stopped')


@app.subscribe('{geo_topic}')
def on_geo(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    if 'geo' not in ctx.app['modalities']:
        return
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    energy = float(np.std(samples))
    thresh: float = ctx.cfg.get('geo_energy_thresh', 0.0)
    if energy < thresh:
        logger.debug('geo energy %.1f below threshold %.1f; dropping window', energy, thresh)
        return
    ts_s = msg.timestamp // 1_000_000_000
    with ctx.app.lock:
        ctx.app['buffer'].add(ctx.cfg['geo_topic'], ts_s, samples)


@app.subscribe('{mic_topic}')
def on_mic(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    if 'mic' not in ctx.app['modalities']:
        return
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    energy = float(np.std(samples))
    thresh: float = ctx.cfg.get('mic_energy_thresh', 0.0)
    if energy < thresh:
        logger.debug('mic energy %.1f below threshold %.1f; dropping window', energy, thresh)
        return
    ts_s = msg.timestamp // 1_000_000_000
    with ctx.app.lock:
        ctx.app['buffer'].add(ctx.cfg['mic_topic'], ts_s, samples)


@app.schedule(1.0)
def run_inference(ctx: AciesContext) -> None:
    modalities: list[str] = ctx.app['modalities']
    _mod_to_topic = {'geo': ctx.cfg['geo_topic'], 'mic': ctx.cfg['mic_topic']}
    keys = [_mod_to_topic[m] for m in modalities]

    try:
        with ctx.app.lock:
            samples = ctx.app['buffer'].pop(keys, INPUT_LEN)
    except ValueError:
        with ctx.app.lock:
            ts_by_topic = {k: sorted(ctx.app['buffer']._data[k]) for k in keys}
        logger.debug('not enough buffered data for inference; timestamps=%s', ts_by_topic)
        return

    # --- build FoundationSense model input ---
    # expected format: {'shake': {'seismic': tensor, 'audio': tensor}}
    data: dict[str, dict[str, torch.Tensor]] = {'shake': {}}
    for mod in modalities:
        topic = _mod_to_topic[mod]
        arr: npt.NDArray[np.float32] = np.concatenate([v for _, v in sorted(samples[topic].items())]).astype(np.float32)
        if mod == 'geo':
            # 2s x 200 Hz = 400 -> downsample x2 -> 200 -> (1, 1, 10, 20)
            data['shake']['seismic'] = torch.from_numpy(arr[::2].reshape(1, 1, 10, 20))  # pyright: ignore[reportUnknownMemberType]
        else:
            # 2s x 16000 Hz = 32000 -> downsample x2 -> 16000 -> (1, 1, 10, 1600)
            data['shake']['audio'] = torch.from_numpy(arr[::2].reshape(1, 1, 10, 1600))  # pyright: ignore[reportUnknownMemberType]

    t0 = time.perf_counter_ns()
    logit, _feat = ctx.app['model'](data)  # returns [[score_0, score_1, ...]]
    infer_ms = (time.perf_counter_ns() - t0) / 1_000_000

    # logit shape: (num_targets, num_classes), values are probabilities
    probs: list[list[float]] = np.array(logit).tolist()
    ensemble_buf: deque[list[list[float]]] = ctx.app['ensemble_buf']
    ensemble_buf.append(probs)
    logger.debug(
        'inference: probs=%s infer_ms=%.1f ensemble=%d',
        [[f'{x:.3f}' for x in row] for row in probs],
        infer_ms,
        len(ensemble_buf),
    )

    if len(ensemble_buf) < ctx.cfg.get('ensemble_size', 1):
        return

    # --- soft-vote ensemble: average probabilities across the window ---
    # shape: (ensemble_win, num_targets, num_classes) -> mean over axis 0
    ensemble_probs: npt.NDArray[np.float64] = np.array(list(ensemble_buf)).mean(axis=0)

    labels: list[str] = ctx.cfg.get('labels') or []
    predictions: list[AciesPrediction] = []
    for target_probs in ensemble_probs:
        for label, score in zip(labels, target_probs):
            if score > 0:
                predictions.append(AciesPrediction(label=label, score=float(score)))

    if len(predictions) > 0:
        msg = AciesInference(source=ctx.ns.base, timestamp=ctx.now(), predictions=predictions)
        logger.debug(f'Inference result: {msg}')
        ctx.publish(ctx.app['output_topic'], msg)


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
