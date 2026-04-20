"""SPAR vehicle classifier + localizer node for AciesOS.

Subscribes to AciesTimeSeries messages on geo and mic topic patterns that fan
out across all nodes (e.g. ``**/geo``, ``**/mic``). Every second, spar takes
all data that landed for the next 2-second window from every node / every
modality, runs a single model pass, and publishes predictions that include
both a class label and a location (lat/lon).

Unlike vibrofm (which aligns one geo + one mic stream per inference), spar
is a global, multi-node model: whatever data arrived for the window gets
fed in, and the model itself decides how to fuse it.

Usage::

    acies-spar --weight /path/to/cls.pt
               [--tracking-weight /path/to/track.pt]
               --geo '**/geo' --mic '**/mic'
               [--output TOPIC]
               [--labels car,truck,person]
               [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import torch
from acies.core import AciesApp, AciesContext, OnChange, setup_logging
from acies.core.msg import AciesInference, AciesKvChange, AciesPrediction, AciesTimeSeries
from acies.SPAR.inference import ModelForInference  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

# samples required for one inference call: N x 1-second windows.
# Observed sample rates on ICT replay: geo=200 Hz, mic=16000 Hz, so a
# 2-second window yields tensors of shape (400,) for geo and (32000,)
# for mic per node.
INPUT_LEN = 2

# seconds to wait after a window's end before firing inference, to let
# late-arriving node messages land in the buffer.
WINDOW_GRACE_S = 1

DEFAULT_LABELS = ['polaris', 'warthog', 'truck', 'husky']


app = AciesApp()


# --- lifecycle -------------------------------------------------------------


@app.on_startup
def setup(ctx: AciesContext) -> None:
    labels: list[str] = ctx.cfg.get('labels') or []
    tracking_weight = ctx.cfg.get('tracking_weight')
    model = ModelForInference(
        weight=Path(ctx.cfg['weight']),
        num_classes=len(labels) or None,
        tracking_weight=Path(tracking_weight) if tracking_weight else None,
    )
    logger.info(
        'loaded model from %s; tracking_weight=%s #classes=%d #params=%d',
        ctx.cfg['weight'],
        tracking_weight,
        model.num_classes,
        sum(p.numel() for p in model.parameters()),
    )

    output_topic: str = ctx.cfg.get('output_topic') or ctx.ns.topic('vehicle')
    ctx.app['model'] = model
    ctx.app['output_topic'] = output_topic
    # buffer: {ts_s: {node: {modality: samples}}}
    ctx.app['buffer'] = {}
    ctx.app['latest_ts_s'] = 0
    ctx.app['next_window_start'] = None

    ctx.cfg['start_at'] = time.time()

    logger.info(
        'publishing to %s; geo=%r mic=%r labels=%s',
        output_topic,
        ctx.cfg.get('geo_topic'),
        ctx.cfg.get('mic_topic'),
        ctx.cfg.get('labels'),
    )


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('spar stopped')


# --- kv change notifications ----------------------------------------------


@app.subscribe(OnChange('start_at'))
def on_start_at_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Clear buffers when a new replay starts so stale data doesn't pollute inference."""
    logger.info('start_at changed to %s; clearing buffers', msg.value)
    with ctx.app.lock:
        ctx.app['buffer'].clear()
        ctx.app['latest_ts_s'] = 0
        ctx.app['next_window_start'] = None


@app.subscribe(OnChange('weight'))
def on_weight_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Reload model weights when the weight file changes."""
    new_weight = msg.value
    tracking_weight = ctx.cfg.get('tracking_weight')
    logger.info('weight changed to %s (tracking=%s); reloading', new_weight, tracking_weight)
    ctx.app['model'].reload(new_weight, tracking_weight)
    with ctx.app.lock:
        ctx.app['buffer'].clear()
        ctx.app['next_window_start'] = None
    logger.info('weight reload complete')


@app.subscribe(OnChange('tracking_weight'))
def on_tracking_weight_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    """Reload both heads when the tracking weight file changes."""
    new_tracking = msg.value
    weight = ctx.cfg['weight']
    logger.info('tracking_weight changed to %s; reloading', new_tracking)
    ctx.app['model'].reload(weight, new_tracking)
    with ctx.app.lock:
        ctx.app['buffer'].clear()
        ctx.app['next_window_start'] = None
    logger.info('tracking_weight reload complete')


@app.subscribe(OnChange('labels'))
def on_labels_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    new_labels = msg.value
    logger.info('labels changed to %s', new_labels)
    ctx.cfg['labels'] = new_labels if isinstance(new_labels, list) else new_labels.split(',')


# --- incoming sensor data -------------------------------------------------


def _node_id(source: str) -> str:
    """Extract a node identifier from a message's source string.

    Sources look like ``rs1/geo`` / ``edge-01/mic``; we take the leading segment.
    """
    return source.split('/', 1)[0] if source else 'unknown'


def _ingest(ctx: AciesContext, modality: str, msg: AciesTimeSeries) -> None:
    samples: npt.NDArray[np.int_] = np.frombuffer(msg.payload[0], dtype=msg.dtype)
    ts_s = msg.timestamp // 1_000_000_000
    node = _node_id(msg.source)
    with ctx.app.lock:
        buf: dict[int, dict[str, dict[str, npt.NDArray[Any]]]] = ctx.app['buffer']
        buf.setdefault(ts_s, {}).setdefault(node, {})[modality] = samples
        if ts_s > ctx.app['latest_ts_s']:
            ctx.app['latest_ts_s'] = ts_s
        if ctx.app['next_window_start'] is None:
            ctx.app['next_window_start'] = ts_s


@app.subscribe('{geo_topic}')
def on_geo(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    _ingest(ctx, 'geo', msg)


@app.subscribe('{mic_topic}')
def on_mic(ctx: AciesContext, msg: AciesTimeSeries) -> None:
    _ingest(ctx, 'mic', msg)


# --- inference ------------------------------------------------------------


@app.schedule(1.0)
def run_inference(ctx: AciesContext) -> None:
    with ctx.app.lock:
        buf: dict[int, dict[str, dict[str, npt.NDArray[Any]]]] = ctx.app['buffer']
        latest: int = ctx.app['latest_ts_s']
        next_start: int | None = ctx.app['next_window_start']

        if next_start is None or not buf:
            return

        window_end = next_start + INPUT_LEN - 1
        if latest < window_end + WINDOW_GRACE_S:
            return  # still waiting on (possibly late) data for this window

        window_secs = list(range(next_start, window_end + 1))
        collected: dict[str, dict[str, list[tuple[int, npt.NDArray[Any]]]]] = {}
        for w in window_secs:
            for node, mods in buf.pop(w, {}).items():
                for modality, samples in mods.items():
                    collected.setdefault(node, {}).setdefault(modality, []).append((w, samples))

        ctx.app['next_window_start'] = next_start + INPUT_LEN

        # drop any stragglers older than the new window start; they can no
        # longer contribute to a future window.
        for old in [t for t in buf if t < ctx.app['next_window_start']]:
            buf.pop(old, None)

    if not collected:
        logger.debug('empty window [%d..%d]; skipping', next_start, window_end)
        return

    # --- build model input: {node: {modality: concatenated tensor}} ---
    data: dict[str, dict[str, torch.Tensor]] = {}
    for node, mods in collected.items():
        data[node] = {}
        for modality, chunks in mods.items():
            arr = np.concatenate([v for _, v in sorted(chunks)]).astype(np.float32)
            data[node][modality] = torch.from_numpy(arr)  # pyright: ignore[reportUnknownMemberType]

    # Log the shape of every per-(node, modality) tensor going into the model
    # so we can design the real model against the actual data layout.
    # shape_report = {
    #     node: {modality: tuple(t.shape) for modality, t in mods.items()}
    #     for node, mods in data.items()
    # }
    # Raw-sample range sanity check: training assumed int16-range counts
    # (mic ~±32768, geo ~16000 DC offset). If any publisher rescales to
    # normalized floats, dB offsets are miscalibrated and predictions silently
    # degrade.
    # stats_report = {
    #     node: {
    #         modality: (
    #             f'min={t.min().item():.2f} max={t.max().item():.2f} '
    #             f'mean={t.mean().item():.2f} std={t.std().item():.2f}'
    #         )
    #         for modality, t in mods.items()
    #     }
    #     for node, mods in data.items()
    # }
    # logger.info(
    #     'spar input window=[%d..%d] nodes=%d shapes=%s stats=%s',
    #     next_start,
    #     window_end,
    #     len(data),
    #     shape_report,
    #     stats_report,
    # )

    t0 = time.perf_counter_ns()
    class_probs, locations = ctx.app['model'](data)
    infer_ms = (time.perf_counter_ns() - t0) / 1_000_000

    probs_2d: npt.NDArray[np.float32] = np.atleast_2d(np.asarray(class_probs, dtype=np.float32))
    labels: list[str] = ctx.cfg.get('labels') or []

    # Scene invariant: exactly one vehicle present, so report only the argmax
    # class rather than the full softmax distribution.
    predictions: list[AciesPrediction] = []
    for target_idx, target_probs in enumerate(probs_2d):
        lat, lon = locations[target_idx] if target_idx < len(locations) else (None, None)
        best_idx = int(np.argmax(target_probs))
        label = labels[best_idx] if best_idx < len(labels) else str(best_idx)
        predictions.append(
            AciesPrediction(
                label=label,
                score=float(target_probs[best_idx]),
                latitude=float(lat) if lat is not None else None,
                longitude=float(lon) if lon is not None else None,
                extras={
                    'infer_ms': infer_ms,
                    'window_start_s': next_start,
                    'nodes': sorted(collected.keys()),
                },
            )
        )

    logger.debug(
        'inference window=[%d..%d] nodes=%d probs=%s locations=%s infer_ms=%.1f',
        next_start,
        window_end,
        len(collected),
        [[f'{x:.3f}' for x in row] for row in probs_2d],
        [
            (f'{lat:.6f}' if lat is not None else None, f'{lon:.6f}' if lon is not None else None)
            for lat, lon in locations
        ],
        infer_ms,
    )

    if predictions:
        out_msg = AciesInference(source=ctx.ns.base, timestamp=ctx.now(), predictions=predictions)
        ctx.publish(ctx.app['output_topic'], out_msg)


# --- CLI ------------------------------------------------------------------


@app.cli()
@click.option('--weight', required=True, type=click.Path(exists=True), help='Classification weight file (backbone + class head).')
@click.option(
    '--tracking-weight',
    default=None,
    type=click.Path(exists=True),
    help='Optional tracking-head weight file; loaded on top of --weight.',
)
@click.option(
    '--geo',
    'geo_topic',
    required=True,
    help='Geo input topic pattern, e.g. **/geo to fan out across all nodes.',
)
@click.option(
    '--mic',
    'mic_topic',
    required=True,
    help='Mic input topic pattern, e.g. **/mic to fan out across all nodes.',
)
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
def main(
    weight: str,
    tracking_weight: str | None,
    geo_topic: str,
    mic_topic: str,
    output_topic: str | None,
    labels: str,
) -> None:
    app.state.config.update(
        {
            'deactivated': False,
            'weight': weight,
            'tracking_weight': tracking_weight,
            'geo_topic': geo_topic,
            'mic_topic': mic_topic,
            'output_topic': output_topic,
            'labels': labels.split(','),
        }
    )
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
