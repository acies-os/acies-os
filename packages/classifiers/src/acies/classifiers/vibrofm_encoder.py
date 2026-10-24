"""VibroFM encoder node for AciesOS.

Subscribes to AciesTimeSeries messages on configured geo, mic, and drone_mic
topics, buffers two 1-second windows per modality, and runs one FoundationSense
encoder every second for two independent pipelines:

- vehicle: geo + mic (the model's modalities) -> vehicle embedding
- drone:   drone_mic as audio-only input      -> drone embedding

Each pipeline has its own buffer, so mic and drone_mic can be active at the
same time. Embeddings are published as AciesEmbedding so that task heads
(see ``vibrofm_task_head``) can subscribe to the pipeline they need.

Modalities required by the vehicle pipeline are determined from the model
config file at startup. --geo, --mic, and --drone-mic topics must be
provided; unused topics are silently ignored. The drone pipeline runs only
if the model includes the audio modality.

Usage::

    acies-vfm-encoder --weight /path/to/encoder.pt
                      --geo <topic> --mic <topic> --drone-mic <topic>
                      [--output TOPIC] [--drone-output TOPIC]
                      [--geo-energy-thresh FLOAT] [--mic-energy-thresh FLOAT]
                      [--drone-mic-energy-thresh FLOAT]
                      [--modality seismic|audio]
                      [--freq-mae] [--deactivated]
                      [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import torch
from acies.buffers import TemporalBuffer
from acies.classifiers.timing import LOG_INTERVAL_S, Timings
from acies.core import AciesApp, AciesContext, OnChange, setup_logging
from acies.core.msg import AciesEmbedding, AciesKvChange, AciesTimeSeries
from acies.FoundationSense.embedding import VibrofmEmbeddingModel  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

# samples required per modality for one inference call: 2 x 1-second windows
INPUT_LEN = 2

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

    # VibrofmEmbeddingModel calls argparse.parse_args() internally at construction;
    # clear sys.argv so it does not see our Click arguments and error out.
    timings = Timings()
    _saved_argv = sys.argv[:]
    sys.argv = sys.argv[:1]
    try:
        with timings.measure('load'):
            model = VibrofmEmbeddingModel(Path(ctx.cfg['weight']), freq_mae, modality=ctx.cfg.get('modality'))
    finally:
        sys.argv = _saved_argv
    ctx.app['timings'] = timings
    raw_mods: list[str] = model.args.dataset_config['modality_names']
    modalities = [_MOD_MAPPING[m] for m in raw_mods]
    logger.info(
        'loaded encoder from %s in %.0f ms; modalities=%s freq_mae=%s #params=%d',
        ctx.cfg['weight'],
        timings.last('load'),
        modalities,
        freq_mae,
        sum(p.numel() for p in model.parameters()),
    )

    # end in /embedding so `**/vehicle` and `**/drone` prediction subscribers don't match
    output_topic: str = ctx.cfg.get('output_topic') or ctx.ns.topic('vehicle', 'embedding')
    drone_output_topic: str = ctx.cfg.get('drone_output_topic') or ctx.ns.topic('drone', 'embedding')
    ctx.app['model'] = model
    ctx.app['modalities'] = modalities
    ctx.app['output_topic'] = output_topic
    ctx.app['drone_output_topic'] = drone_output_topic
    ctx.app['buffer'] = TemporalBuffer(size=INPUT_LEN + 2)
    ctx.app['drone_buffer'] = TemporalBuffer(size=INPUT_LEN + 2)

    ctx.cfg['start_at'] = time.time()

    is_deactivated = ctx.cfg.get('deactivated', False)
    ctx.app.config.setdefault('sys', {})['state'] = 'deactivated' if is_deactivated else 'active'

    logger.info(
        'publishing vehicle to %s, drone to %s; geo_thresh=%.1f mic_thresh=%.1f drone_mic_thresh=%.1f',
        output_topic,
        drone_output_topic,
        ctx.cfg.get('geo_energy_thresh', 0.0),
        ctx.cfg.get('mic_energy_thresh', 0.0),
        ctx.cfg.get('drone_mic_energy_thresh', 0.0),
    )


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    logger.info('vibrofm encoder stopped; timing: %s', ctx.app['timings'].format())


@app.schedule(LOG_INTERVAL_S)
def log_timings(ctx: AciesContext) -> None:
    """Log rolling load/reload/inference timings and keep them in sys/timing (not published)."""
    timings: Timings = ctx.app['timings']
    ctx.app.config.setdefault('sys', {})['timing'] = timings.summary()
    logger.info('timing: %s', timings.format())


@app.subscribe(OnChange('start_at'))
def on_start_at_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Clear buffers when a new replay starts so stale data doesn't pollute inference."""
    logger.info('start_at changed to %s; clearing buffers', msg.value)
    with ctx.app.lock:
        ctx.app['buffer'].clear()
        ctx.app['drone_buffer'].clear()


@app.subscribe(OnChange('weight'))
def on_weight_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Reload encoder weights and update modalities when the weight file changes."""
    new_weight = msg.value
    modality_cfg = ctx.cfg.get('modality')
    logger.info('weight changed to %s (modality=%s); reloading', new_weight, modality_cfg)
    timings: Timings = ctx.app['timings']
    with timings.measure('reload'):
        ctx.app['model'].reload(new_weight, modality_cfg)
    ctx.app['modalities'] = [_MOD_MAPPING[m] for m in ctx.app['model'].args.dataset_config['modality_names']]

    with ctx.app.lock:
        ctx.app['buffer'].clear()
        ctx.app['drone_buffer'].clear()
    logger.info('weight reload complete in %.0f ms; modalities=%s', timings.last('reload'), ctx.app['modalities'])


@app.subscribe(OnChange('deactivated'))
def on_deactivated_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    is_deactivated = msg.value
    logger.info('deactivated changed to %s', is_deactivated)
    ctx.app.config.setdefault('sys', {})['state'] = 'deactivated' if is_deactivated else 'active'


def _add_window(ctx: AciesContext, msg: AciesTimeSeries, name: str, buffer: str) -> None:
    """Energy-gate one 1-second window and add it to the given buffer under its topic."""
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    energy = float(np.std(samples))
    thresh: float = ctx.cfg.get(f'{name}_energy_thresh', 0.0)
    if energy < thresh:
        logger.debug('%s energy %.1f below threshold %.1f; dropping window', name, energy, thresh)
        return
    ts_s = msg.timestamp // 1_000_000_000
    with ctx.app.lock:
        ctx.app[buffer].add(ctx.cfg[f'{name}_topic'], ts_s, samples)


@app.subscribe('{geo_topic}')
def on_geo(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    if 'geo' in ctx.app['modalities']:
        _add_window(ctx, msg, 'geo', 'buffer')


@app.subscribe('{mic_topic}')
def on_mic(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    if 'mic' in ctx.app['modalities']:
        _add_window(ctx, msg, 'mic', 'buffer')


@app.subscribe('{drone_mic_topic}')
def on_drone_mic(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    if 'mic' in ctx.app['modalities']:
        _add_window(ctx, msg, 'drone_mic', 'drone_buffer')


def _encode(ctx: AciesContext, pipeline: str, buffer: str, mod_to_topic: dict[str, str], output_topic: str) -> None:
    """Pop INPUT_LEN aligned windows for mod_to_topic from buffer, encode them, and publish the embedding."""
    keys = list(mod_to_topic.values())
    try:
        with ctx.app.lock:
            samples = ctx.app[buffer].pop(keys, INPUT_LEN)
    except ValueError:
        with ctx.app.lock:
            ts_by_topic = {k: sorted(ctx.app[buffer]._data[k]) for k in keys}
        logger.debug('%s: not enough buffered data for inference; timestamps=%s', pipeline, ts_by_topic)
        return

    # --- build FoundationSense model input ---
    # expected format: {'shake': {'seismic': tensor, 'audio': tensor}}; audio-only omits 'seismic'
    data: dict[str, dict[str, torch.Tensor]] = {'shake': {}}
    energy: dict[str, float] = {}
    for mod, topic in mod_to_topic.items():
        arr: npt.NDArray[np.float32] = np.concatenate([v for _, v in sorted(samples[topic].items())]).astype(np.float32)
        energy[mod] = float(np.std(arr))
        if mod == 'geo':
            # 2s x 200 Hz = 400 -> downsample x2 -> 200 -> (1, 1, 10, 20)
            data['shake']['seismic'] = torch.from_numpy(arr[::2].reshape(1, 1, 10, 20))  # pyright: ignore[reportUnknownMemberType]
        else:
            # 2s x 16000 Hz = 32000 -> downsample x2 -> 16000 -> (1, 1, 10, 1600)
            data['shake']['audio'] = torch.from_numpy(arr[::2].reshape(1, 1, 10, 1600))  # pyright: ignore[reportUnknownMemberType]

    t0 = time.perf_counter_ns()
    feat = ctx.app['model'](data)  # features concatenated across input modalities
    infer_ms = (time.perf_counter_ns() - t0) / 1_000_000
    ctx.app['timings'].record(f'infer_{pipeline}', infer_ms)  # infer_vehicle / infer_drone

    # feat shape: (num_targets, embed_dim)
    feat_arr: npt.NDArray[np.float32] = np.ascontiguousarray(feat, dtype=np.float32)
    logger.debug('%s embedding: shape=%s infer_ms=%.1f', pipeline, feat_arr.shape, infer_ms)

    extras: dict[str, Any] = {'infer_ms': infer_ms}
    if 'geo' in energy:
        extras['geo_energy'] = energy['geo']
    if 'mic' in energy:
        extras['mic_energy'] = energy['mic']
    msg = AciesEmbedding(
        source=ctx.ns.base,
        timestamp=ctx.now(),
        payload=feat_arr.tobytes(),
        shape=list(feat_arr.shape),
        dtype=str(feat_arr.dtype),
        extras=extras,
    )
    ctx.publish(output_topic, msg)


@app.schedule(1.0)
def run_inference(ctx: AciesContext) -> None:
    if ctx.cfg.get('deactivated'):
        logger.debug('vibrofm encoder deactivated; skipping inference')
        return

    modalities: list[str] = ctx.app['modalities']
    _mod_to_topic = {'geo': ctx.cfg['geo_topic'], 'mic': ctx.cfg['mic_topic']}

    # vehicle: all model modalities from geo + mic
    _encode(ctx, 'vehicle', 'buffer', {m: _mod_to_topic[m] for m in modalities}, ctx.app['output_topic'])

    # drone: audio-only from drone_mic, independent of the vehicle pipeline
    if 'mic' in modalities:
        _encode(ctx, 'drone', 'drone_buffer', {'mic': ctx.cfg['drone_mic_topic']}, ctx.app['drone_output_topic'])


@app.cli()
@click.option('--weight', required=True, type=click.Path(exists=True), help='Encoder weight file path.')
@click.option('--geo', 'geo_topic', required=True, help='Geo input topic (AciesTimeSeries).')
@click.option('--mic', 'mic_topic', required=True, help='Vehicle mic input topic (AciesTimeSeries).')
@click.option('--drone-mic', 'drone_mic_topic', required=True, help='Drone mic input topic (AciesTimeSeries).')
@click.option(
    '--output',
    'output_topic',
    default=None,
    help='Output topic for vehicle AciesEmbedding results. Defaults to <host>/<name>/vehicle/embedding.',
)
@click.option(
    '--drone-output',
    'drone_output_topic',
    default=None,
    help='Output topic for drone AciesEmbedding results. Defaults to <host>/<name>/drone/embedding.',
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
    '--drone-mic-energy-thresh',
    default=0.0,
    type=float,
    show_default=True,
    help='Drone mic std threshold; windows below are dropped.',
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
@click.option(
    '--deactivated',
    is_flag=True,
    default=False,
    show_default=True,
    help='Start deactivated; inference begins once the deactivated key is set to false.',
)
def main(
    weight: str,
    geo_topic: str,
    mic_topic: str,
    drone_mic_topic: str,
    output_topic: str | None,
    drone_output_topic: str | None,
    geo_energy_thresh: float,
    mic_energy_thresh: float,
    drone_mic_energy_thresh: float,
    modality: str | None,
    freq_mae: bool,
    deactivated: bool,
) -> None:
    app.state.config.update(
        {
            'deactivated': deactivated,
            'weight': weight,
            'geo_topic': geo_topic,
            'mic_topic': mic_topic,
            'drone_mic_topic': drone_mic_topic,
            'output_topic': output_topic,
            'drone_output_topic': drone_output_topic,
            'geo_energy_thresh': geo_energy_thresh,
            'mic_energy_thresh': mic_energy_thresh,
            'drone_mic_energy_thresh': drone_mic_energy_thresh,
            'modality': modality,
            'freq_mae': freq_mae,
        }
    )
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
