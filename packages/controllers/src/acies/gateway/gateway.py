"""WebSocket gateway node for AciesOS.

Bridges the internal Zenoh mesh to a browser UI over WebSocket.

Outbound (Zenoh -> browser):
    Subscribes to configured Zenoh topics and forwards each message to all
    connected WebSocket clients as a WsFrame(topic, payload).

Inbound (browser -> Zenoh):
    Receives WsFrame messages from the browser and dispatches them to the
    appropriate Zenoh control services.

Usage::

    acies-gateway --config gq.toml
                  [--acies-listen=ws://0.0.0.0:8765]
                  [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import glob
import logging
import os
import random
from collections import defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import click
import numpy as np
import tomli as tomllib
from acies.buffers.temporal import TimeWindow
from acies.core import AciesApp, AciesContext, setup_logging
from acies.core.ctl import io_call, kv_call, kv_set
from acies.core.msg import (
    AciesHeartbeat,
    AciesInference,
    AciesIoResponse,
    AciesPrediction,
    AciesResult,
    Err,
    KvEntry,
)
from acies.core.namespace import matches

from .graph import build_io_graph

logger = logging.getLogger(__name__)

# suppress verbose logs from websockets library used by the WebSocketTransport
ws_logger = logging.getLogger('websockets')
ws_logger.setLevel(logging.CRITICAL)
ws_logger.propagate = False

app = AciesApp()

_NS_PER_S = 1_000_000_000
_DEFAULT_ENSEMBLE_WIN_S = 30
_DEFAULT_HEARTBEAT_INTERVAL_S = 5
_DEFAULT_ALIVE_MULTIPLIER = 3  # service is alive if last heartbeat < interval * multiplier
_REPLAY_AHEAD_TIME_S = 30
_IO_QUERY_INTERVAL_S = 10
_IO_QUERY_TIMEOUT_S = 2.0  # must stay well below the interval so runs do not overlap
_IO_QUERY_MAX_WORKERS = 16
_IO_QUERY_STATES = frozenset({'active'})  # only apps in these states are queried
_GRAPH_PUBLISH_INTERVAL_S = 2  # re-publish so a newly connected UI does not wait for the next io query
_MODALITY_MAP = {
    'seismic': 'geo',
    'acoustic': 'mic',
    'both': 'both',
}
_ALL_MODELS = ['vfm', 'diffphys', 'spar', 'vfm_encoder', 'vfm_vehicle', 'vfm_drone', 'vfm_drone_e2e']
_MODEL_NAME_MAP: dict[str, str] = {'vibrofm': 'vfm', 'spar': 'spar', 'diffphys': 'diffphys'}
_SPAR_NAMESPACE = 'joint'

_MODALITY_TO_WEIGHT_KEY: dict[str, str] = {'both': 'both', 'geo': 'seismic', 'mic': 'audio'}

# VibroFM pipelines selectable per node in the UI: 'e2e' runs the monolithic `vfm`
# service; 'fm' runs the shared `vfm_encoder` + `vfm_vehicle` task head.
_DEFAULT_PIPELINE = 'e2e'
# confidence_threshold table to use for each prediction source service
_THRESHOLD_MODEL: dict[str, str] = {'vfm_vehicle': 'vfm', 'vfm_drone_e2e': 'vfm_drone'}


@app.subscribe('**/spar')
def on_spar(ctx: AciesContext, msg: AciesInference) -> None:
    logger.debug('tk-on_spar: msg.predictions: %s', msg.predictions)
    thresholds: dict[str, dict[str, float]] = ctx.app['confidence_threshold']
    model_thresh = thresholds.get('spar', {})
    filtered_predictions = [p for p in msg.predictions if p.score >= model_thresh.get(p.label, 0.0)]
    logger.debug('tk-on_spar: filtered_predictions: %s', filtered_predictions)
    if filtered_predictions:
        msg_spar = AciesInference(source=msg.source, timestamp=msg.timestamp, predictions=filtered_predictions)
        ctx.publish('ws://spar', msg_spar)


@app.subscribe('**/vehicle')
def on_vehicle(ctx: AciesContext, msg: AciesInference) -> None:
    for pred in msg.predictions:
        logger.debug('from %s: %s', msg.source, pred)
    with ctx.app.lock:
        ctx.app['ensemble_buf'].append((msg.timestamp, msg.predictions))
    # Filter by per-model confidence thresholds before forwarding to UI
    model = msg.source.rsplit('/', 1)[-1]
    thresholds: dict[str, dict[str, float]] = ctx.app['confidence_threshold']
    model_thresh = thresholds.get(_THRESHOLD_MODEL.get(model, model), {})
    filtered = [p for p in msg.predictions if p.score >= model_thresh.get(p.label, 0.0)]
    if filtered:
        msg_filtered = AciesInference(source=msg.source, timestamp=msg.timestamp, predictions=filtered)
        ctx.publish('ws://predictions', msg_filtered)


@app.subscribe('**/drone')
def on_drone(ctx: AciesContext, msg: AciesInference) -> None:
    """Forward drone classifications (vfm_drone head or vfm_drone_e2e) to the UI, apart from vehicle predictions."""
    model = msg.source.rsplit('/', 1)[-1]
    thresholds: dict[str, dict[str, float]] = ctx.app['confidence_threshold']
    model_thresh = thresholds.get(_THRESHOLD_MODEL.get(model, model), {})
    filtered = [p for p in msg.predictions if p.score >= model_thresh.get(p.label, 0.0)]
    if filtered:
        ctx.publish('ws://drone', AciesInference(source=msg.source, timestamp=msg.timestamp, predictions=filtered))


if os.getenv('ACIES_GATEWAY_DEBUG'):

    @app.subscribe('**/energy')
    def on_energy(ctx: AciesContext, msg: Any) -> None:
        energy_by_ch: dict[str, float] = msg['energy']
        energy = next(iter(energy_by_ch.values()))
        source: str = msg['source']

        # normalize against rolling 95th percentile over a 60s window
        wins: dict[str, TimeWindow] = ctx.task.data.setdefault('energy_wins', {})
        if source not in wins:
            wins[source] = TimeWindow(window_ns=60 * _NS_PER_S, data_clock=True)
        win = wins[source]
        win.add(source, msg['timestamp'], energy)
        values = np.fromiter((v for _, v in win.get(source)), dtype=np.float64)
        p95 = float(np.percentile(values, 95)) if values.size > 0 else energy
        normalized = min(energy / p95, 1.0) if p95 > 0 else 0.0

        ctx.publish('ws://energy', {'source': source, 'timestamp': msg['timestamp'], 'energy': energy})
        logger.debug('energy from %s: %.2f (p95=%.2f normalized=%.3f)', source, energy, p95, normalized)

    @app.schedule(10)
    def print_energy_wins(ctx: AciesContext) -> None:
        wins: dict[str, TimeWindow] = ctx.task.data.get('energy_wins', {})
        for source, win in wins.items():
            logger.debug('energy win %s: %s', source, win)


@app.subscribe('**/ctl/heartbeat')
def on_heartbeat(ctx: AciesContext, msg: AciesHeartbeat) -> None:
    # --- validate source format (must contain namespace/name) ---
    if '/' not in msg.source:
        logger.warning('heartbeat source %r missing namespace/name separator; dropping', msg.source)
        return

    buff: TimeWindow = ctx.app['heartbeat']
    buff.add(msg.source, msg.timestamp, msg.state)
    logger.debug('heartbeat from %s: state=%s ts=%d', msg.source, msg.state, msg.timestamp)


def _find_services(heartbeat_buf: TimeWindow, host: str) -> dict[str, str]:
    """Find all services on a host from heartbeat records.

    Returns {name: full_source_path}, e.g. {'mic': 'rs1/mic', 'geo': 'rs1/geo'}.
    The host is matched against the first segment of each source path.
    """
    result: dict[str, str] = {}
    for source in heartbeat_buf.keys():
        first_seg = source.split('/', 1)[0]
        if first_seg == host:
            name = source.rsplit('/', 1)[1]
            result[name] = source
    return result


def _wait_futures(
    items: Sequence[tuple[str, Sequence[KvEntry], Future[list[AciesResult]]]],
    label: str,
) -> None:
    """Wait for all futures to complete, logging per-op results per target."""
    for target, ops, f in items:
        try:
            results = f.result()
        except Exception:
            logger.exception('%s: %s: request failed', label, target)
            continue
        if not results:
            logger.warning('%s: %s: no response (timeout)', label, target)
            continue
        parts: list[str] = []
        for op, result in zip(ops, results):
            key_str = '/'.join(op.key)
            if isinstance(result, Err):
                parts.append(f'{key_str}=Err({result.reason})')
            else:
                parts.append(f'{key_str}=Ok')
        logger.info('%s: %s: %s', label, target, ' '.join(parts))


def _resolve_route(map_cfg: dict[str, Any], target_list: list[str]) -> tuple[str, int, str] | None:
    """Resolve a target list to (scene, run_id, reconfig_target) from the map config."""
    reconfig_target = '_'.join(sorted(t.lower() for t in target_list))
    routes = map_cfg.get('routes', {})
    if reconfig_target not in routes:
        logger.error('reconfig target %s not found in routes', reconfig_target)
        return None
    scene, run_id = tuple(routes[reconfig_target].split('/'))
    return str(scene), int(run_id.removeprefix('run')), reconfig_target


def _build_node_states(
    msg_nodes: list[dict[str, Any]],
    map_node_mapping: dict[str, str],
    scene: str,
    run_id: int,
) -> dict[str, dict[str, str | int | None]]:
    """Build per-node state dict from the UI message node list.

    ``pipeline`` ('fm' | 'e2e') and ``drone`` (drone name or None) are only sent
    for single-node identification; they default to the monolithic VibroFM and
    no drone.
    """
    new_node_states: dict[str, dict[str, str | int | None]] = {}
    for node_state in msg_nodes:
        node_id = node_state['nodeId'].lower()
        mapped_node_id: str = map_node_mapping.get(node_id, node_id)
        model = _MODEL_NAME_MAP.get(node_state['model'].lower(), node_state['model'].lower())
        modality = _MODALITY_MAP.get(node_state['modality'].lower(), node_state['modality'].lower())
        new_node_states[mapped_node_id] = {
            'node_id': mapped_node_id,
            'replayed_node_id': node_id,
            'model': model,
            'modality': modality,
            'scene': scene,
            'run_id': run_id,
            'pipeline': (node_state.get('pipeline') or _DEFAULT_PIPELINE).lower(),
            'drone': node_state.get('drone'),
        }
    return new_node_states


def _drone_truth(node_states: dict[str, dict[str, str | int | None]], map_cfg: dict[str, Any]) -> dict[str, str | None]:
    """Drone each node was told to replay on ``drone_mic`` (None when off), mirroring ``_build_reconfig_ops``."""
    runs: dict[str, Any] = map_cfg.get('models', {}).get('drone', {}).get('runs', {})
    spar_active = any(state['model'] == 'spar' for state in node_states.values())
    return {
        node_id: None if spar_active or state['drone'] not in runs else str(state['drone'])
        for node_id, state in node_states.items()
    }


def _build_fm_ops(
    map_cfg: dict[str, Any],
    services: dict[str, str],
    node_id: str,
    use_fm: bool,
    drone: str | None,
    drone_fm: bool,
) -> tuple[list[tuple[str, Sequence[KvEntry]]], list[tuple[str, Sequence[KvEntry]]], list[str]]:
    """Build KV ops for the split VibroFM pipeline, the drone classifier, and drone audio replay on one node.

    ``use_fm`` runs vehicle classification on the shared encoder + ``vfm_vehicle`` head. A drone is
    classified by the encoder + ``vfm_drone`` head when ``drone_fm`` (the node's 'fm' pipeline) and by
    the end-to-end ``vfm_drone_e2e`` model otherwise. The encoder runs whenever one of its heads does.
    All models are already deactivated by the generic model deactivation pass, so only activation is
    emitted here.

    Returns (deactivate_pass, data_pass, targets).
    """
    deactivate_pass: list[tuple[str, Sequence[KvEntry]]] = []
    data_pass: list[tuple[str, Sequence[KvEntry]]] = []
    targets: list[str] = []
    models_cfg: dict[str, Any] = map_cfg.get('models', {})
    drone_cfg: dict[str, Any] = models_cfg.get('drone', {})

    if drone is not None and drone not in drone_cfg.get('runs', {}):
        logger.error(
            '%s: drone %r not in [models.drone.runs] (available: %s)', node_id, drone, list(drone_cfg.get('runs', {}))
        )
        drone = None

    # --- drone audio replay: on with the selected drone's run, off otherwise ---
    if 'drone_mic' in services:
        deactivate_pass.append((services['drone_mic'], [kv_set('deactivated', value=drone is None)]))
        if drone is not None:
            drone_data_ops: list[KvEntry] = [
                kv_set('scene', value=drone_cfg.get('scene', 'audrok')),
                kv_set('run', value=int(drone_cfg['runs'][drone])),
                kv_set('node', value=drone_cfg.get('node', 'rs1')),
            ]
            data_pass.append((services['drone_mic'], drone_data_ops))
            targets.append(services['drone_mic'])
    elif drone is not None:
        logger.warning('%s: drone %r selected but no drone_mic service found in heartbeat', node_id, drone)

    drone_head = drone is not None and drone_fm
    drone_e2e = drone is not None and not drone_fm

    # --- shared encoder ---
    if (use_fm or drone_head) and 'vfm_encoder' in services:
        encoder_weight: str = models_cfg.get('vfm_encoder', {}).get('weight', '')
        data_pass.append(
            (services['vfm_encoder'], [kv_set('deactivated', value=False), kv_set('weight', value=encoder_weight)])
        )
        targets.append(services['vfm_encoder'])
    elif use_fm or drone_head:
        logger.warning('%s: fm pipeline selected but no vfm_encoder service found in heartbeat', node_id)

    # --- task heads ---
    vfm_cfg: dict[str, Any] = models_cfg.get('vfm', {})
    vehicle_cfg = {'head': vfm_cfg.get('fm', {}).get('head', ''), 'labels': vfm_cfg.get('labels')}
    heads: list[tuple[str, dict[str, Any], bool]] = [
        ('vfm_vehicle', vehicle_cfg, use_fm),
        ('vfm_drone', drone_cfg, drone_head),
    ]
    for head, cfg, active in heads:
        if not active:
            continue
        if head not in services:
            logger.warning('%s: %s selected but no %s service found in heartbeat', node_id, head, head)
            continue
        head_weight: str = cfg.get('head', '')
        head_ops: list[KvEntry] = [
            kv_set('deactivated', value=False),
            kv_set('weight', value=head_weight),
            kv_set('labels', value=cfg.get('labels')),
        ]
        data_pass.append((services[head], head_ops))
        targets.append(services[head])

    # --- end-to-end drone model (full AuDroK checkpoint, audio only) ---
    if drone_e2e and 'vfm_drone_e2e' in services:
        e2e_ops: list[KvEntry] = [
            kv_set('deactivated', value=False),
            kv_set('modality', value='audio'),
            kv_set('weight', value=drone_cfg.get('weight', '')),
            kv_set('labels', value=drone_cfg.get('labels')),
        ]
        data_pass.append((services['vfm_drone_e2e'], e2e_ops))
        targets.append(services['vfm_drone_e2e'])
    elif drone_e2e:
        logger.warning('%s: end-to-end drone selected but no vfm_drone_e2e service found in heartbeat', node_id)

    return deactivate_pass, data_pass, targets


def _build_reconfig_ops(
    node_states: dict[str, dict[str, str | int | None]],
    heartbeat_buf: TimeWindow,
    map_cfg: dict[str, Any],
    edge_services: dict[str, str],
    scene: str,
    run_id: int,
    reconfig_target: str,
) -> tuple[list[tuple[str, Sequence[KvEntry]]], list[tuple[str, Sequence[KvEntry]]], list[str]]:
    """Build all KV operation passes for the reconfiguration.

    Returns (deactivate_pass, data_pass, all_targets).
    """
    deactivate_pass: list[tuple[str, Sequence[KvEntry]]] = []
    data_pass: list[tuple[str, Sequence[KvEntry]]] = []
    all_targets: list[str] = [edge_services['gps']]
    tracker = edge_services.get('tracker')
    if tracker is not None:
        all_targets.append(tracker)

    # Collect which models are active across all nodes
    active_models: set[str] = {str(state['model']) for state in node_states.values()}
    spar_active = 'spar' in active_models

    # --- per-node: sensor activation + model activation/deactivation ---
    for node_id, state in node_states.items():
        services = _find_services(heartbeat_buf, node_id)
        selected_model: str = str(state['model'])
        modality: str = str(state['modality'])
        # SPAR needs all geo/mic across all nodes; force 'both' when active
        effective_modality: str = 'both' if spar_active else modality

        logger.debug(
            'node_id: %s, selected_model: %s, modality: %s, effective_modality: %s',
            node_id,
            selected_model,
            modality,
            effective_modality,
        )

        # --- sensor deactivation/activation ---
        if effective_modality not in ('geo', 'both') and 'geo' in services:
            deactivate_pass.append((services['geo'], [kv_set('deactivated', value=True)]))
        if effective_modality not in ('mic', 'both') and 'mic' in services:
            deactivate_pass.append((services['mic'], [kv_set('deactivated', value=True)]))
        if effective_modality in ('geo', 'both') and 'geo' in services:
            deactivate_pass.append((services['geo'], [kv_set('deactivated', value=False)]))
        if effective_modality in ('mic', 'both') and 'mic' in services:
            deactivate_pass.append((services['mic'], [kv_set('deactivated', value=False)]))

        # --- model deactivation: deactivate models NOT selected on this node ---
        for model_name in _ALL_MODELS:
            if model_name in services:
                deactivate_pass.append((services[model_name], [kv_set('deactivated', value=True)]))

        # --- sensor data config ---
        data_ops: list[KvEntry] = [
            kv_set('scene', value=state['scene']),
            kv_set('run', value=state['run_id']),
            kv_set('node', value=state['replayed_node_id']),
        ]
        if effective_modality in ('mic', 'both') and 'mic' in services:
            data_pass.append((services['mic'], data_ops))
            all_targets.append(services['mic'])
        if effective_modality in ('geo', 'both') and 'geo' in services:
            data_pass.append((services['geo'], data_ops))
            all_targets.append(services['geo'])

        # --- VibroFM split pipeline (encoder + task heads) and drone add-on ---
        use_fm = selected_model == 'vfm' and state['pipeline'] == 'fm'
        drone = None if spar_active else state['drone']
        fm_deactivate, fm_data, fm_targets = _build_fm_ops(
            map_cfg, services, node_id, use_fm, None if drone is None else str(drone), state['pipeline'] == 'fm'
        )
        deactivate_pass.extend(fm_deactivate)
        data_pass.extend(fm_data)
        all_targets.extend(fm_targets)

        # --- per-node model activation + config (vfm e2e, diffphys) ---
        if selected_model != 'spar' and not use_fm and selected_model in services:
            model_cfg = map_cfg.get('models', {}).get(selected_model, {})
            model_weights: dict[str, str] = model_cfg.get('weight', {})
            model_labels: list[str] | None = model_cfg.get('labels')
            weight_key: str = _MODALITY_TO_WEIGHT_KEY.get(effective_modality.lower(), 'both')
            model_ops: list[KvEntry] = [
                kv_set('deactivated', value=False),
                kv_set('weight', value=model_weights[weight_key]),
                kv_set('labels', value=model_labels),
            ]
            # Send modality override so the model updates its expected modalities
            if effective_modality == 'geo':
                model_ops.append(kv_set('modality', value='seismic'))
            elif effective_modality == 'mic':
                model_ops.append(kv_set('modality', value='audio'))
            else:
                model_ops.append(kv_set('modality', value=None))
            data_pass.append((services[selected_model], model_ops))
            all_targets.append(services[selected_model])

    # --- SPAR global activation ---
    spar_services = _find_services(heartbeat_buf, _SPAR_NAMESPACE)
    spar_cfg = map_cfg.get('models', {}).get('spar', {})
    if 'spar' in spar_services:
        # spar always consumes geo+mic, so select the 'both' entry. One
        # unified weight file per scene covers backbone + classification
        # + localization heads (vehicle_classification_tracking task).

        if spar_active:
            spar_weight: str = spar_cfg.get('weight', {}).get('both', '')
            spar_ops: list[KvEntry] = [
                kv_set('deactivated', value=False),
                kv_set('weight', value=spar_weight),
                kv_set('labels', value=spar_cfg.get('labels')),
            ]
        else:
            spar_ops = [kv_set('deactivated', value=True)]
        data_pass.append((spar_services['spar'], spar_ops))
        all_targets.append(spar_services['spar'])
    else:
        logger.warning('spar selected but no spar service found in heartbeat at %s', _SPAR_NAMESPACE)

    # GPS service config
    data_pass.append(
        (
            edge_services['gps'],
            [
                kv_set('scene', value=scene),
                kv_set('run', value=run_id),
                kv_set('label', value=reconfig_target),
            ],
        )
    )

    logger.debug('build_output: deactivate_pass: %s', deactivate_pass)
    logger.debug('build_output: data_pass: %s', data_pass)
    logger.debug('build_output: all_targets: %s', all_targets)

    return deactivate_pass, data_pass, all_targets


@app.subscribe('ws://ctl')
def on_ctl(ctx: AciesContext, msg: Any) -> None:
    logger.debug('ctl command received: %r', msg.get('appType', 'unknown'))
    reconfig_map = msg['map']

    # Look up the config for the selected map
    configs: dict[str, dict[str, Any]] = ctx.app['configs']
    if reconfig_map not in configs:
        logger.error('unknown map: %s (available: %s)', reconfig_map, list(configs.keys()))
        return
    map_cfg = configs[reconfig_map]

    # Swap active config state on map change
    active_map: str | None = ctx.app.get('active_map')
    if active_map != reconfig_map:
        ctx.app['gps'] = map_cfg.get('gps', {})
        ctx.app['confidence_threshold'] = map_cfg.get('confidence_threshold', {})
        ctx.app['map_node_mapping'] = map_cfg.get('map_node_mapping', {})
        ctx.app['active_map'] = reconfig_map
        ctx.app['ensemble_buf'].clear()
        logger.info('map changed: %s -> %s', active_map, reconfig_map)

    # Resolve route
    route = _resolve_route(map_cfg, msg['target'])
    if route is None:
        return
    scene, run_id, reconfig_target = route

    # Build per-node state
    node_states = _build_node_states(msg['nodes'], map_cfg.get('map_node_mapping', {}), scene, run_id)
    with ctx.app.lock:
        ctx.app['drone_truth'] = _drone_truth(node_states, map_cfg)
    logger.debug('new replay config: %s', node_states)

    # Discover edge services
    heartbeat_buf: TimeWindow = ctx.app['heartbeat']
    edge_host = ctx.ns.namespace.split('/')[0]
    edge_services = _find_services(heartbeat_buf, edge_host)
    if 'gps' not in edge_services:
        logger.error('gps service not found in heartbeat records')
        return

    # Build KV operation passes
    deactivate_pass, data_pass, all_targets = _build_reconfig_ops(
        node_states,
        heartbeat_buf,
        map_cfg,
        edge_services,
        scene,
        run_id,
        reconfig_target,
    )

    # Execute in three passes
    with ThreadPoolExecutor() as pool:
        # this is the deactivation pass that controls what runs on each nodes
        # this should send to mic/geo/vfm/diffphys on each node
        items0 = [(t, ops, pool.submit(kv_call, ctx, t, ops)) for t, ops in deactivate_pass]
        _wait_futures(items0, 'pass 0 (deactivation)')

        items1 = [(t, ops, pool.submit(kv_call, ctx, t, ops)) for t, ops in data_pass]
        _wait_futures(items1, 'pass 1 (data params)')

        start_at = float(ctx.now() / _NS_PER_S) + _REPLAY_AHEAD_TIME_S
        start_ops: Sequence[KvEntry] = [kv_set('start_at', value=start_at)]
        items2 = [(t, start_ops, pool.submit(kv_call, ctx, t, start_ops)) for t in all_targets]
        _wait_futures(items2, 'pass 2 (start_at)')


def get_north_and_south_end(gps: dict[str, list[float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Returns the north and south end of the GPS coordinates as (lat, lon) tuples."""
    north_lat = float(max(coord[0] for coord in gps.values()))
    south_lat = float(min(coord[0] for coord in gps.values()))
    north_lon = float(np.mean([coord[1] for coord in gps.values()]))
    south_lon = float(np.mean([coord[1] for coord in gps.values()]))
    return (north_lat, north_lon), (south_lat, south_lon)


@app.schedule(1.0)
def model_metrics(ctx: AciesContext) -> None:
    # TODO: replace with real metrics tracking and calculation
    ctx.publish(
        'ws://performance',
        {'rs1': {'f1': min(random.gauss(0.8, 0.05), 1.0), 'accuracy': min(random.gauss(0.7, 0.05), 1.0)}},
    )


# @app.schedule(1)
def dummy_gps(ctx: AciesContext) -> None:
    # TODO: replace with GPS data
    # - subscribe to a GPS topic (from replay)?
    # - Add MQTT support to obtain real-time GPS data
    north, south = get_north_and_south_end(ctx.app['gps'])
    # t oscillates 0 -> 1 (south->north) -> 0 (north->south), step 0.05
    t: float = ctx.get('t', 0.0)
    direction: int = ctx.get('direction', 1)
    lat = south[0] + t * (north[0] - south[0])
    lon = south[1] + t * (north[1] - south[1])
    noise_lat = random.gauss(0, 0.00001)
    noise_lon = random.gauss(0, 0.00001)
    ctx.publish('ws://gps_truth', {'suv': {'lat': lat, 'lon': lon}})
    ctx.publish('ws://gps', {'suv': {'lat': lat + noise_lat, 'lon': lon + noise_lon}})
    t += direction / 15.0
    if t >= 1.0:
        t, direction = 1.0, -1
    elif t <= 0.0:
        t, direction = 0.0, 1
    ctx['t'] = t
    ctx['direction'] = direction


@app.subscribe('**/gps/truth', '**/gps')
def on_gps(ctx: AciesContext, msg: Any, topic: str) -> None:
    ts_ns = msg.pop('timestamp', -1)
    if matches('**/gps/truth', topic):
        ctx.publish('ws://gps_truth', msg)
        logger.debug('gps update: %r t=%.2f', msg, float(ts_ns / _NS_PER_S) if ts_ns else None)
    else:
        ctx.publish('ws://gps', msg)
        logger.debug('gps estimate: %r t=%.2f', msg, float(ts_ns / _NS_PER_S) if ts_ns else None)


@app.schedule(1.0)
def system_health(ctx: AciesContext) -> None:
    heartbeat_buf: TimeWindow = ctx.app['heartbeat']
    gps_table: dict[str, list[float]] = ctx.app['gps']
    now = ctx.now()

    heartbeat_interval = ctx.cfg.get('heartbeat_interval', _DEFAULT_HEARTBEAT_INTERVAL_S)
    alive_timeout_ns = ctx.cfg.get('alive_timeout', heartbeat_interval * _DEFAULT_ALIVE_MULTIPLIER) * _NS_PER_S

    hosts: dict[str, dict[str, Any]] = {}
    for source in heartbeat_buf.keys():
        # skip infrastructure services (e.g. edge/infra/gateway)
        if matches('**/infra/**', source):
            continue
        entry = heartbeat_buf.latest(source)
        if entry is None:
            continue
        ts, state = entry
        alive = (now - ts) < alive_timeout_ns

        # source is "namespace/name" -> group by first segment (host)
        parts = source.split('/', 1)
        host = parts[0]
        map_node_mapping: dict[str, str] = ctx.app.get('map_node_mapping', {})
        reverse_mapping = {v: k for k, v in map_node_mapping.items()}
        gps_key = reverse_mapping.get(host, host)
        # skip hosts not in the current map's GPS table (only when mapping is defined)
        if map_node_mapping and gps_key not in gps_table:
            continue
        if host not in hosts:
            coords = gps_table.get(gps_key, [])
            hosts[host] = {
                'services': [],
                'lat': coords[0] if len(coords) > 0 else None,
                'lon': coords[1] if len(coords) > 1 else None,
            }
        hosts[host]['services'].append(
            {
                'name': source,
                'state': state if alive else 'down',
                'alive': alive,
                'last_heartbeat': ts,
            }
        )

    # ground truth for the drone classifier: the drone this host's drone_mic is replaying
    with ctx.app.lock:
        drone_truth: dict[str, str | None] = dict(ctx.app['drone_truth'])
    for host, info in hosts.items():
        replaying = any(s['name'] == f'{host}/drone_mic' and s['state'] == 'active' for s in info['services'])
        info['drone'] = drone_truth.get(host) if replaying else None

    logger.debug('system health: %s ', hosts)
    ctx.publish('ws://health', {'map': ctx.app.get('active_map'), 'hosts': hosts})


def _store_io_cache(ctx: AciesContext, io_cache: dict[str, AciesIoResponse]) -> None:
    """Replace the io cache and the graph derived from it."""
    graph = build_io_graph(io_cache)
    with ctx.app.lock:
        ctx.app['io_cache'] = io_cache
        ctx.app['graph'] = graph


@app.schedule(_IO_QUERY_INTERVAL_S)
def collect_io(ctx: AciesContext) -> None:
    """Query ctl/io of every live node and cache the raw responses.

    Result: ``ctx.app['io_cache']``, a ``{source: AciesIoResponse}`` dict that
    is replaced as a whole each run, and ``ctx.app['graph']`` built from it.
    A node that fails to answer keeps its
    previous response (``response.timestamp`` tells how old it is) until its
    heartbeat is no longer alive or its state is filtered out.
    """
    heartbeat_buf: TimeWindow = ctx.app['heartbeat']
    now = ctx.now()
    heartbeat_interval = ctx.cfg.get('heartbeat_interval', _DEFAULT_HEARTBEAT_INTERVAL_S)
    alive_timeout_ns = ctx.cfg.get('alive_timeout', heartbeat_interval * _DEFAULT_ALIVE_MULTIPLIER) * _NS_PER_S

    targets: list[str] = []
    for source in heartbeat_buf.keys():
        entry = heartbeat_buf.latest(source)
        if entry is None:
            continue
        ts, state = entry
        if (now - ts) < alive_timeout_ns and state in _IO_QUERY_STATES:
            targets.append(source)

    if not targets:
        _store_io_cache(ctx, {})
        return

    with ThreadPoolExecutor(max_workers=min(len(targets), _IO_QUERY_MAX_WORKERS)) as pool:
        futures = {t: pool.submit(io_call, ctx, t, timeout=_IO_QUERY_TIMEOUT_S) for t in targets}

    old: dict[str, AciesIoResponse] = ctx.app['io_cache']
    new: dict[str, AciesIoResponse] = {}
    for target, f in futures.items():
        try:
            resp = f.result()
        except Exception:
            logger.exception('io query to %s failed', target)
            resp = None
        resp = resp or old.get(target)
        if resp is not None:
            new[target] = resp

    _store_io_cache(ctx, new)
    logger.debug('io cache: %d/%d node(s) answered', len(new), len(targets))


@app.schedule(_GRAPH_PUBLISH_INTERVAL_S)
def publish_graph(ctx: AciesContext) -> None:
    """Publish the latest I/O graph to the UI; the UI filters it by selection."""
    with ctx.app.lock:
        graph = ctx.app['graph']
    ctx.publish('ws://graph', {'timestamp': ctx.now(), **graph})


@app.schedule(1.0)
def run_ensemble(ctx: AciesContext) -> None:
    window_ns: int = ctx.cfg.get('ensemble_win', _DEFAULT_ENSEMBLE_WIN_S) * _NS_PER_S
    now = ctx.now()
    cutoff = now - window_ns

    with ctx.app.lock:
        buf: deque[tuple[int, list[AciesPrediction]]] = ctx.app['ensemble_buf']
        # prune entries older than the window
        while buf and buf[0][0] < cutoff:
            _ = buf.popleft()
        entries = list(buf)

    if not entries:
        return

    # --- average scores per label across all sources in the window ---
    label_scores: dict[str, list[float]] = defaultdict(list)
    for _ts, preds in entries:
        for pred in preds:
            label_scores[pred.label].append(pred.score)

    logger.debug('ensemble t=%d, entires=%d', now, len(entries))
    predictions: list[AciesPrediction] = []
    for label, scores in label_scores.items():
        avg = sum(scores) / len(scores)
        if avg > 0:
            pred = AciesPrediction(label=label, score=avg)
            predictions.append(pred)
            logger.debug('  - %s', pred)


def _reload_config(ctx: AciesContext) -> None:
    """Read the toml config and update app state derived from it."""
    with open(ctx.cfg['config_path'], 'rb') as f:
        ctx.cfg.update(tomllib.load(f))
    ctx.app['gps'] = ctx.cfg.get('gps', {})
    ctx.app['confidence_threshold'] = ctx.cfg.get('confidence_threshold', {})
    ctx.app['map_node_mapping'] = ctx.cfg.get('map_node_mapping', {})
    ctx.app['config_mtime'] = os.path.getmtime(ctx.cfg['config_path'])

    # Refresh the matching entry in the pre-loaded configs dict
    config_name = os.path.splitext(os.path.basename(ctx.cfg['config_path']))[0]
    configs: dict[str, dict[str, Any]] | None = ctx.app.data.get('configs')
    if configs is not None:
        configs[config_name] = dict(ctx.cfg)  # snapshot current merged config


@app.on_startup
def setup(ctx: AciesContext) -> None:
    _reload_config(ctx)

    # Pre-load all TOML configs from the same directory, keyed by filename stem
    config_dir = os.path.dirname(ctx.cfg['config_path'])
    configs: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(f'{config_dir}/*.toml')):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as f:
            configs[name] = tomllib.load(f)
    ctx.app['configs'] = configs
    ctx.app['active_map'] = None
    logger.info('loaded %d config(s): %s', len(configs), list(configs.keys()))

    logger.info('confidence thresholds: %d model(s)', len(ctx.app['confidence_threshold']))
    ctx.app['ensemble_buf'] = deque()
    ctx.app['drone_truth'] = {}  # {node: drone replayed on its drone_mic}, set on each deploy
    heartbeat_interval = ctx.cfg.get('heartbeat_interval', _DEFAULT_HEARTBEAT_INTERVAL_S)
    alive_timeout = ctx.cfg.get('alive_timeout', heartbeat_interval * _DEFAULT_ALIVE_MULTIPLIER)
    # Keep heartbeat history for at least 2x the alive timeout
    ctx.app['heartbeat'] = TimeWindow(int(alive_timeout * 2) * _NS_PER_S)
    ctx.app['io_cache'] = {}  # source -> latest AciesIoResponse, filled by collect_io
    ctx.app['graph'] = build_io_graph({})  # derived from io_cache, published on ws://graph
    logger.info(
        'gateway ready: %d node(s) in gps table, ensemble_win=%ds, alive_timeout=%ds',
        len(ctx.app['gps']),
        ctx.cfg.get('ensemble_win', 5),
        alive_timeout,
    )


@app.schedule(5.0)
def check_config(ctx: AciesContext) -> None:
    """Reload the toml config file if it has been modified on disk."""
    try:
        mtime = os.path.getmtime(ctx.cfg['config_path'])
    except OSError:
        return
    if mtime != ctx.app['config_mtime']:
        logger.info('config file changed on disk; reloading %s', ctx.cfg['config_path'])
        _reload_config(ctx)


@app.on_shutdown
def teardown(_ctx: AciesContext) -> None:
    logger.info('gateway stopped')


@app.cli()
@click.option(
    '--config',
    'config_path',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Site config file (.toml).',
)
@click.option(
    '--ensemble-win',
    default=30,
    type=int,
    show_default=True,
    help='Ensemble window size in seconds.',
)
def main(**kwargs: dict[str, Any]) -> None:
    app.state.config.update(kwargs)
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
