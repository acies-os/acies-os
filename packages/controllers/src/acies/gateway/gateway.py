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

import logging
import os
import random
from collections import defaultdict, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import click
import numpy as np
import tomli as tomllib
from acies.buffers.temporal import TimeWindow
from acies.corev2 import AciesApp, AciesContext, setup_logging
from acies.corev2.ctl import kv_call, kv_set
from acies.corev2.msg import AciesHeartbeat, AciesInference, AciesPrediction, KvEntry
from acies.corev2.namespace import matches

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


@app.subscribe('**/energy')
def on_energy(ctx: AciesContext, msg: Any) -> None:
    energy_by_ch: dict[str, float] = msg['energy']
    energy = next(iter(energy_by_ch.values()))
    ctx.publish('ws://energy', {'source': msg['source'], 'timestamp': msg['timestamp'], 'energy': energy})
    logger.debug('energy from %s: %.2f', msg['source'], energy)


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


def _wait_futures(futures: list[Any], label: str) -> None:
    """Wait for all futures to complete, logging errors."""
    for f in futures:
        try:
            f.result()
        except Exception:
            logger.exception('%s: kv request failed', label)
    logger.info('%s complete: %d target(s)', label, len(futures))


@app.subscribe('ws://ctl')
def on_ctl(ctx: AciesContext, msg: Any) -> None:
    logger.debug('ctl command received: %r', msg.get('appType', 'unknown'))
    reconfig_map = msg['map']
    reconfig_target_list: list[str] = sorted([t.lower() for t in msg['target']])

    # map selected map and target to the corresponding runID path
    reconfig_target = '_'.join(reconfig_target_list)
    if reconfig_map not in ctx.cfg['routes']:
        logger.error('reconfig map %s not found in routes', reconfig_map)
        return

    if reconfig_target not in ctx.cfg['routes'][reconfig_map]:
        logger.error('reconfig target %s not found in routes', reconfig_target)
        return

    # 2024-08-06-GQ/run29
    reconfig_route = ctx.cfg['routes'][reconfig_map][reconfig_target]
    scene, run_id = tuple(reconfig_route.split('/'))
    run_id = int(run_id.removeprefix('run'))
    scene = str(scene)

    # reconfig node states
    reconfig_node_states: list[dict[str, str]] = msg['nodes']
    new_node_states: dict[str, dict[str, str | int]] = {}
    map_node_mapping = ctx.cfg['map_node_mapping']
    for node_state in reconfig_node_states:
        node_id = node_state['nodeId'].lower()
        mapped_node_id: str = map_node_mapping.get(node_id, node_id)
        # TODO: support changing model
        model = node_state['model'].replace('VibroFM', 'vfm')
        modality = node_state['modality'].lower()
        modality = _MODALITY_MAP.get(modality, modality)

        new_node_states[mapped_node_id] = {
            'node_id': mapped_node_id,  # mapped node id, same for ICT, gq-X mapped to rsY for GCQ
            'replayed_node_id': node_id,  # original node id, used for path routing
            'model': model,  # model name
            'modality': modality,  # modality name [both, seismic, acoustic]
            'scene': scene,  # scene name, e.g. 2024-08-06-GQ
            'run_id': run_id,  # run id, e.g. 29
        }

    logger.debug('new replay config: %s', new_node_states)

    heartbeat_buf: TimeWindow = ctx.app['heartbeat']

    # --- look up GPS service from heartbeat records ---
    edge_host = ctx.ns.namespace.split('/')[0]
    gps_base = _find_services(heartbeat_buf, edge_host).get('gps')
    if gps_base is None:
        logger.error('gps service not found in heartbeat records')
        return

    # --- build kv ops for each target using service discovery ---
    data_reconfig_pass: list[tuple[str, Sequence[KvEntry]]] = []
    all_targets: list[str] = [gps_base]
    for node_id, state in new_node_states.items():
        services = _find_services(heartbeat_buf, node_id)
        data_ops = [
            kv_set('scene', value=state['scene']),
            kv_set('run', value=state['run_id']),
            kv_set('node', value=state['replayed_node_id']),
        ]
        if 'vfm' in services:
            all_targets.append(services['vfm'])
        if state['modality'] in ['mic', 'both'] and 'mic' in services:
            data_reconfig_pass.append((services['mic'], data_ops))
            all_targets.append(services['mic'])
        if state['modality'] in ['geo', 'both'] and 'geo' in services:
            data_reconfig_pass.append((services['geo'], data_ops))
            all_targets.append(services['geo'])
    data_reconfig_pass.append(
        (
            gps_base,
            [
                kv_set('scene', value=scene),
                kv_set('run', value=run_id),
                kv_set('label', value=reconfig_target),
            ],
        )
    )

    with ThreadPoolExecutor() as pool:
        # pass 1: send data params in parallel
        futures = [pool.submit(kv_call, ctx, target, ops) for target, ops in data_reconfig_pass]
        _wait_futures(futures, 'pass 1 (data params)')

        # pass 2: send start_at to all nodes in parallel
        start_at: float = float(ctx.now() / _NS_PER_S) + _REPLAY_AHEAD_TIME_S
        futures = [pool.submit(kv_call, ctx, target, [kv_set('start_at', value=start_at)]) for target in all_targets]
        _wait_futures(futures, 'pass 2 (start_at)')


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


@app.subscribe('**/gps/truth')
def on_gps(ctx: AciesContext, msg: Any) -> None:
    ctx.publish('ws://gps_truth', msg)
    logger.debug('gps update: %r', msg)


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
        if host not in hosts:
            coords = gps_table.get(host, [])
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
    ctx.publish('ws://health', hosts)


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

    logger.info('ensemble t=%d, entires=%d', now, len(entries))
    predictions: list[AciesPrediction] = []
    for label, scores in label_scores.items():
        avg = sum(scores) / len(scores)
        if avg > 0:
            pred = AciesPrediction(label=label, score=avg)
            predictions.append(pred)
            logger.info('  - %s', pred)


def _reload_config(ctx: AciesContext) -> None:
    """Read the toml config and update app state derived from it."""
    with open(ctx.cfg['config_path'], 'rb') as f:
        ctx.cfg.update(tomllib.load(f))
    ctx.app['gps'] = ctx.cfg.get('gps', {})
    ctx.app['confidence_threshold'] = ctx.cfg.get('confidence_threshold', {})
    ctx.app['config_mtime'] = os.path.getmtime(ctx.cfg['config_path'])


@app.on_startup
def setup(ctx: AciesContext) -> None:
    _reload_config(ctx)
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
