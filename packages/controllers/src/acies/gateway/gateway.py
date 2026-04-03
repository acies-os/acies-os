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
                  [--acies-host HOST] [--acies-name NAME]
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from typing import Any

import click
import numpy as np
import tomli as tomllib
from acies.buffers.temporal import TimeWindow
from acies.corev2 import AciesApp, AciesContext, setup_logging
from acies.corev2.msg import AciesHeartbeat, AciesInference, AciesPrediction

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


@app.subscribe('**/vehicle')
def on_vehicle(ctx: AciesContext, msg: AciesInference) -> None:
    for pred in msg.predictions:
        logger.debug('from %s: %s', msg.source, pred)
    with ctx.app.lock:
        ctx.app['ensemble_buf'].append((msg.timestamp, msg.predictions))
    ctx.publish('ws://predictions', msg)


@app.subscribe('**/heartbeat')
def on_heartbeat(ctx: AciesContext, msg: AciesHeartbeat) -> None:
    if msg.source == ctx.ns.base:
        return

    # --- validate source format (must be host/name) ---
    if '/' not in msg.source:
        logger.warning('heartbeat source %r missing host/name separator; dropping', msg.source)
        return

    buff: TimeWindow = ctx.app['heartbeat']
    buff.add(msg.source, msg.timestamp, msg.state)
    logger.debug('heartbeat from %s: state=%s ts=%d', msg.source, msg.state, msg.timestamp)


@app.subscribe('ws://ctl')
def on_ctl(_ctx: AciesContext, msg: Any) -> None:
    logger.info('ctl command received: %r', msg)


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


@app.schedule(1)
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


@app.schedule(1.0)
def system_health(ctx: AciesContext) -> None:
    heartbeat_buf: TimeWindow = ctx.app['heartbeat']
    gps_table: dict[str, list[float]] = ctx.app['gps']
    now = ctx.now()

    heartbeat_interval = ctx.cfg.get('heartbeat_interval', _DEFAULT_HEARTBEAT_INTERVAL_S)
    alive_timeout_ns = ctx.cfg.get('alive_timeout', heartbeat_interval * _DEFAULT_ALIVE_MULTIPLIER) * _NS_PER_S

    hosts: dict[str, dict[str, Any]] = {}
    for source in heartbeat_buf.keys():
        entry = heartbeat_buf.latest(source)
        if entry is None:
            continue
        ts, state = entry
        alive = (now - ts) < alive_timeout_ns

        # source is "host/name" -> group by host
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


@app.on_startup
def setup(ctx: AciesContext) -> None:
    with open(ctx.cfg['config_path'], 'rb') as f:
        ctx.cfg.update(tomllib.load(f))
    gps: dict[str, list[float]] = ctx.cfg.get('gps', {})
    confidence_threshold: dict[str, dict[str, float]] = ctx.cfg.get('confidence_threshold', {})
    ctx.app['gps'] = gps
    ctx.app['confidence_threshold'] = confidence_threshold
    ctx.app['ensemble_buf'] = deque()
    heartbeat_interval = ctx.cfg.get('heartbeat_interval', _DEFAULT_HEARTBEAT_INTERVAL_S)
    alive_timeout = ctx.cfg.get('alive_timeout', heartbeat_interval * _DEFAULT_ALIVE_MULTIPLIER)
    # Keep heartbeat history for at least 2x the alive timeout
    ctx.app['heartbeat'] = TimeWindow(int(alive_timeout * 2) * _NS_PER_S)
    logger.info(
        'gateway ready: %d node(s) in gps table, ensemble_win=%ds, alive_timeout=%ds',
        len(gps),
        ctx.cfg.get('ensemble_win', 5),
        alive_timeout,
    )


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
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
