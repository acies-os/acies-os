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

    acies-spar --weight /path/to/spar.pt
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

# Known scene IDs. A scene ID must match a yaml file under
# ``acies/SPAR/vatt/config/{scene}.yaml`` (bundled with the SPAR package).
# We identify the active scene by substring-matching one of these against a
# weight filename.
_KNOWN_SCENES: tuple[str, ...] = ('2024-03-29-ICT', '2026-04-14-West-Point')


def _scene_from_path(path: str) -> str | None:
    name = Path(path).name
    for scene in _KNOWN_SCENES:
        if scene in name:
            return scene
    return None


def _num_classes_for(labels: list[str]) -> int | None:
    """Size the classifier head = len(labels) + 1.

    Every scene reserves the last class slot for an internal "background"
    class that is absent from ``labels`` in .env / *.toml. Predictions whose
    argmax lands on that slot are dropped before publishing.
    """
    if not labels:
        return None
    return len(labels) + 1


app = AciesApp()


# --- lifecycle -------------------------------------------------------------


@app.on_startup
def setup(ctx: AciesContext) -> None:
    labels: list[str] = ctx.cfg.get('labels') or []
    # At startup scene is the SPAR default (ICT); on_weight_change will later
    # rebuild for the scene encoded in the replay weight filename.
    startup_scene = _scene_from_path(ctx.cfg.get('weight', '')) or '2024-03-29-ICT'
    model = ModelForInference(
        weight=Path(ctx.cfg['weight']),
        scene=startup_scene,
        num_classes=_num_classes_for(labels),
    )
    logger.info(
        'loaded model from %s; scene=%s #classes=%d #params=%d',
        ctx.cfg['weight'],
        startup_scene,
        model.num_classes,
        sum(p.numel() for p in model.parameters()),
    )

    output_topic: str = ctx.cfg.get('output_topic') or ctx.ns.topic('spar')
    ctx.app['model'] = model
    ctx.app['output_topic'] = output_topic
    # buffer: {ts_s: {node: {modality: samples}}}
    ctx.app['buffer'] = {}
    ctx.app['latest_ts_s'] = 0
    ctx.app['next_window_start'] = None

    ctx.cfg['start_at'] = time.time()
    
    is_deactivated = ctx.cfg.get('deactivated', False)
    ctx.app.config.setdefault('sys', {})['state'] = 'deactivated' if is_deactivated else 'active'

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
    """Rebuild the model from scratch for the scene encoded in the weight filename.

    The value is a path whose basename carries the scene ID (e.g.
    ``spar_2024-03-29-ICT.pt``). Different scenes can have different
    architectures, so we instantiate a fresh ``ModelForInference`` rather
    than reloading state_dict into the existing backbone.
    """
    new_weight = str(msg.value)
    scene = _scene_from_path(new_weight)
    if scene is None:
        logger.error(
            'on_weight_change: no known scene %s matches %s; ignoring',
            _KNOWN_SCENES, new_weight,
        )
        return

    labels: list[str] = ctx.cfg.get('labels') or []
    num_classes = _num_classes_for(labels)
    logger.info(
        'rebuilding spar model: scene=%s weight=%s #classes=%s',
        scene, new_weight, num_classes,
    )
    new_model = ModelForInference(
        weight=Path(new_weight),
        scene=scene,
        num_classes=num_classes,
    )
    logger.info(
        'rebuilt spar model: scene=%s #classes=%d #params=%d',
        scene, new_model.num_classes,
        sum(p.numel() for p in new_model.parameters()),
    )

    with ctx.app.lock:
        ctx.app['model'] = new_model
        ctx.app['buffer'].clear()
        ctx.app['latest_ts_s'] = 0
        ctx.app['next_window_start'] = None

@app.subscribe(OnChange('deactivated'))
def on_deactivated_change(ctx: AciesContext, msg: AciesKvChange) -> None:
    is_deactivated = msg.value
    logger.info('deactivated changed to %s', is_deactivated)
    ctx.app.config.setdefault('sys', {})['state'] = 'deactivated' if is_deactivated else 'active'

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
    if ctx.cfg.get('deactivated'):
        logger.debug('spar deactivated; skipping inference')
        return

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

    model = ctx.app['model']
    t0 = time.perf_counter_ns()
    class_probs, locations = model(data)
    infer_ms = (time.perf_counter_ns() - t0) / 1_000_000

    probs_2d: npt.NDArray[np.float32] = np.atleast_2d(np.asarray(class_probs, dtype=np.float32))
    labels: list[str] = ctx.cfg.get('labels') or []

    # Every scene reserves the last class index for an internal "background"
    # slot (unified vehicle_classification_tracking convention). Drop the
    # prediction entirely when argmax lands there — no label, no lat/lon.
    bg_idx = model.num_classes - 1

    # Scene invariant: exactly one vehicle present, so report only the argmax
    # class rather than the full softmax distribution.
    predictions: list[AciesPrediction] = []
    for target_idx, target_probs in enumerate(probs_2d):
        lat, lon = locations[target_idx] if target_idx < len(locations) else (None, None)
        best_idx = int(np.argmax(target_probs))
        if best_idx == bg_idx:
            logger.debug(
                'dropping background prediction: window=[%d..%d] probs=%s',
                next_start, window_end, [f'{x:.3f}' for x in target_probs],
            )
            continue
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
@click.option('--weight', required=True, type=click.Path(exists=True), help='Unified spar weight file (backbone + class + localization heads).')
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
    help='Output topic for AciesInference results. Defaults to <host>/<name>.',
)
@click.option(
    '--labels',
    default=','.join(DEFAULT_LABELS),
    show_default=True,
    help='Comma-separated class names matching model output order (excluding the internal background slot).',
)
def main(
    weight: str,
    geo_topic: str,
    mic_topic: str,
    output_topic: str | None,
    labels: str,
) -> None:
    app.state.config.update(
        {
            'deactivated': True,
            'weight': weight,
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
