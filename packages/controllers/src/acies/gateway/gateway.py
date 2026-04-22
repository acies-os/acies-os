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
from acies.core.ctl import kv_call, kv_set
from acies.core.msg import AciesHeartbeat, AciesInference, AciesPrediction, AciesResult, Err, KvEntry
from acies.core.namespace import matches

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
_MODALITY_MAP = {
    'seismic': 'geo',
    'acoustic': 'mic',
    'both': 'both',
}
_ALL_MODELS = ['vfm', 'diffphys', 'spar']
_MODEL_NAME_MAP: dict[str, str] = {'vibrofm': 'vfm', 'spar': 'spar', 'diffphys': 'diffphys'}
_SPAR_NAMESPACE = 'joint'

_MODALITY_TO_WEIGHT_KEY: dict[str, str] = {'both': 'both', 'geo': 'seismic', 'mic': 'audio'}


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
    model_thresh = thresholds.get(model, {})
    filtered = [p for p in msg.predictions if p.score >= model_thresh.get(p.label, 0.0)]
    if filtered:
        msg_filtered = AciesInference(source=msg.source, timestamp=msg.timestamp, predictions=filtered)
        ctx.publish('ws://predictions', msg_filtered)


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


def _resolve_route(
    map_cfg: dict[str, Any], target_list: list[str]
) -> tuple[str, int, str] | None:
    """Resolve a target list to (scene, run_id, reconfig_target) from the map config."""
    reconfig_target = '_'.join(sorted(t.lower() for t in target_list))
    routes = map_cfg.get('routes', {})
    if reconfig_target not in routes:
        logger.error('reconfig target %s not found in routes', reconfig_target)
        return None
    scene, run_id = tuple(routes[reconfig_target].split('/'))
    return str(scene), int(run_id.removeprefix('run')), reconfig_target


def _build_node_states(
    msg_nodes: list[dict[str, str]],
    map_node_mapping: dict[str, str],
    scene: str,
    run_id: int,
) -> dict[str, dict[str, str | int]]:
    """Build per-node state dict from the UI message node list."""
    new_node_states: dict[str, dict[str, str | int]] = {}
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
        }
    return new_node_states


def _build_reconfig_ops(
    node_states: dict[str, dict[str, str | int]],
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
        
        logger.debug('node_id: %s, selected_model: %s, modality: %s, effective_modality: %s', node_id, selected_model, modality, effective_modality)

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

        # --- per-node model activation + config (vfm, diffphys) ---
        if selected_model != 'spar' and selected_model in services:
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
    if spar_active:
        spar_services = _find_services(heartbeat_buf, _SPAR_NAMESPACE)
        spar_cfg = map_cfg.get('models', {}).get('spar', {})
        if 'spar' in spar_services:
            # spar always consumes geo+mic, so select the 'both' entry. One
            # unified weight file per scene covers backbone + classification
            # + localization heads (vehicle_classification_tracking task).
            spar_weight: str = spar_cfg.get('weight', {}).get('both', '')
            spar_ops: list[KvEntry] = [
                kv_set('deactivated', value=False),
                kv_set('weight', value=spar_weight),
                kv_set('labels', value=spar_cfg.get('labels')),
            ]
            data_pass.append((spar_services['spar'], spar_ops))
            all_targets.append(spar_services['spar'])
        else:
            logger.warning('spar selected but no spar service found in heartbeat at %s', _SPAR_NAMESPACE)

    # GPS service config
    data_pass.append((
        edge_services['gps'],
        [
            kv_set('scene', value=scene),
            kv_set('run', value=run_id),
            kv_set('label', value=reconfig_target),
        ],
    ))
    
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
        node_states, heartbeat_buf, map_cfg, edge_services, scene, run_id, reconfig_target,
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

    logger.debug('system health: %s ', hosts)
    ctx.publish('ws://health', {'map': ctx.app.get('active_map'), 'hosts': hosts})


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
    heartbeat_interval = ctx.cfg.get('heartbeat_interval', _DEFAULT_HEARTBEAT_INTERVAL_S)
    alive_timeout = ctx.cfg.get('alive_timeout', heartbeat_interval * _DEFAULT_ALIVE_MULTIPLIER)
    # Keep heartbeat history for at least 2x the alive timeout
    ctx.app['heartbeat'] = TimeWindow(int(alive_timeout * 2) * _NS_PER_S)
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
