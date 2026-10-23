"""VibroFM task head node for AciesOS.

Subscribes to AciesEmbedding messages published by a VibroFM encoder
(see ``vibrofm_encoder``) and runs a lightweight FoundationSense task head on
each embedding. Raw inference results are accumulated in a rolling ensemble
buffer; the soft-voted result is published once enough results are available.

Several task heads can subscribe to the same encoder output topic.

Usage::

    acies-vfm-head --weight /path/to/class_layer.pt
                   --input <embedding topic>
                   [--output TOPIC]
                   [--labels car,truck,person]
                   [--ensemble-win N] [--ensemble-size N]
                   [--embed-dim N]
                   [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import torch
from acies.core import AciesApp, AciesContext, OnChange, setup_logging
from acies.core.msg import AciesEmbedding, AciesInference, AciesKvChange, AciesPrediction
from acies.FoundationSense.task_head import VibrofmTaskHead  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

DEFAULT_LABELS = ['polaris', 'warthog', 'truck', 'husky']


app = AciesApp()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    ensemble_win: int = ctx.cfg.get('ensemble_win', 1)

    model = VibrofmTaskHead(Path(ctx.cfg['weight']), embed_dim=ctx.cfg.get('embed_dim', 256))
    logger.info(
        'loaded task head from %s; embed_dim=%d num_classes=%d',
        ctx.cfg['weight'],
        model.embed_dim,
        model.num_classes,
    )

    output_topic: str = ctx.cfg.get('output_topic') or ctx.ns.topic('vehicle')
    ctx.app['model'] = model
    ctx.app['output_topic'] = output_topic
    ctx.app['ensemble_buf'] = deque(maxlen=ensemble_win)

    ctx.cfg['start_at'] = time.time()

    is_deactivated = ctx.cfg.get('deactivated', False)
    ctx.app.config.setdefault('sys', {})['state'] = 'deactivated' if is_deactivated else 'active'

    logger.info(
        'subscribed to %s; publishing to %s; ensemble_win=%d ensemble_size=%d',
        ctx.cfg['input_topic'],
        output_topic,
        ensemble_win,
        ctx.cfg.get('ensemble_size', 1),
    )


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('vibrofm task head stopped')


@app.subscribe(OnChange('start_at'))
def on_start_at_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Clear the ensemble when a new replay starts so stale results don't pollute the vote."""
    logger.info('start_at changed to %s; clearing ensemble buffer', msg.value)
    with ctx.app.lock:
        ctx.app['ensemble_buf'].clear()


@app.subscribe(OnChange('weight'))
def on_weight_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Reload task head weights when the weight file changes."""
    new_weight = msg.value
    logger.info('weight changed to %s; reloading', new_weight)
    ctx.app['model'].reload(new_weight)

    with ctx.app.lock:
        ctx.app['ensemble_buf'].clear()
    logger.info('weight reload complete; num_classes=%d', ctx.app['model'].num_classes)


@app.subscribe(OnChange('deactivated'))
def on_deactivated_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    is_deactivated = msg.value
    logger.info('deactivated changed to %s', is_deactivated)
    ctx.app.config.setdefault('sys', {})['state'] = 'deactivated' if is_deactivated else 'active'


@app.subscribe(OnChange('labels'))
def on_labels_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Update class labels when the label list changes."""
    new_labels = msg.value
    logger.info('labels changed to %s', new_labels)
    ctx.cfg['labels'] = new_labels if isinstance(new_labels, list) else new_labels.split(',')


@app.subscribe('{input_topic}')
def on_embedding(ctx: AciesContext, msg: AciesEmbedding) -> None:
    if ctx.cfg.get('deactivated'):
        logger.debug('vibrofm task head deactivated; skipping inference')
        return

    feat: npt.NDArray[np.float32] = np.frombuffer(msg.payload, dtype=msg.dtype).reshape(msg.shape)
    embed_dim: int = ctx.app['model'].embed_dim
    if feat.shape[-1] != embed_dim:
        # e.g. a multimodal encoder emits 2 x 256 features but the head expects 256
        logger.warning('embedding dim %d does not match task head embed_dim %d; dropping', feat.shape[-1], embed_dim)
        return

    t0 = time.perf_counter_ns()
    logit = ctx.app['model'](torch.from_numpy(feat.astype(np.float32)))  # pyright: ignore[reportUnknownMemberType]
    head_ms = (time.perf_counter_ns() - t0) / 1_000_000

    # logit shape: (num_targets, num_classes), values are probabilities
    probs: list[list[float]] = np.array(logit).tolist()
    with ctx.app.lock:
        ensemble_buf: deque[list[list[float]]] = ctx.app['ensemble_buf']
        ensemble_buf.append(probs)
        logger.debug(
            'inference: probs=%s head_ms=%.1f ensemble=%d',
            [[f'{x:.3f}' for x in row] for row in probs],
            head_ms,
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
                # carry encoder metadata (infer_ms, geo/mic energy) through to the prediction
                extras: dict[str, Any] = {**msg.extras, 'head_ms': head_ms}
                target_pred = AciesPrediction(label=label, score=float(score), extras=extras)
                predictions.append(target_pred)
                logger.debug('%s', target_pred)

    if len(predictions) > 0:
        out = AciesInference(source=ctx.ns.base, timestamp=ctx.now(), predictions=predictions)
        ctx.publish(ctx.app['output_topic'], out)


@app.cli()
@click.option('--weight', required=True, type=click.Path(exists=True), help='Task head weight file path.')
@click.option('--input', 'input_topic', required=True, help='Encoder embedding topic (AciesEmbedding).')
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
    '--ensemble-win',
    default=1,
    type=int,
    show_default=True,
    help='Rolling window size (number of embeddings) for soft-vote ensemble.',
)
@click.option(
    '--ensemble-size', default=1, type=int, show_default=True, help='Minimum results in window before publishing.'
)
@click.option(
    '--embed-dim',
    default=256,
    type=int,
    show_default=True,
    help='Embedding dimension produced by the encoder.',
)
def main(
    weight: str,
    input_topic: str,
    output_topic: str | None,
    labels: str,
    ensemble_win: int,
    ensemble_size: int,
    embed_dim: int,
) -> None:
    app.state.config.update(
        {
            'deactivated': False,
            'weight': weight,
            'input_topic': input_topic,
            'output_topic': output_topic,
            'labels': labels.split(','),
            'ensemble_win': ensemble_win,
            'ensemble_size': ensemble_size,
            'embed_dim': embed_dim,
        }
    )
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
