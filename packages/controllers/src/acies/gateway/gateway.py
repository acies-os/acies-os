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
from acies.corev2 import AciesApp, AciesContext, setup_logging
from acies.corev2.msg import AciesInference, AciesPrediction

logger = logging.getLogger(__name__)

app = AciesApp()

_NS_PER_S = 1_000_000_000
_DEFAULT_ENSEMBLE_WIN_S = 30


@app.subscribe('**/vehicle')
def on_vehicle(ctx: AciesContext, msg: AciesInference) -> None:
    for pred in msg.predictions:
        logger.debug('from %s: %s', msg.source, pred)
    with ctx.app.lock:
        ctx.app['ensemble_buf'].append((msg.timestamp, msg.predictions))
    ctx.publish('ws://predictions', msg)


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
    ctx.publish(
        'ws://performance',
        {'rs1': {'f1': min(random.gauss(0.8, 0.05), 1.0), 'accuracy': min(random.gauss(0.7, 0.05), 1.0)}},
    )


@app.schedule(1)
def dummy_gps(ctx: AciesContext) -> None:
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
def dummy_health(ctx: AciesContext) -> None:
    system_health = {
        'rs1': {
            'servicies': [
                'rs1/geo',
                'rs1/mic',
                'rs1/vfm',
            ],
            'lat': 40.2887754,
            'lon': -88.1261283,
        }
    }
    ctx.publish('ws://health', system_health)


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
    logger.info(
        'gateway ready: %d node(s) in gps table, ensemble_win=%ds',
        len(gps),
        ctx.cfg.get('ensemble_win', 5),
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
