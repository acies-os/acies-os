"""Experiment recorder for acies-os network emulation.

Writes one JSON line per message to ``~/.acies/emu/records/<UTC time>.jsonl``:

* ``spar``   -- every SPAR output on ``**/spar`` (label, score, lat/lon, data
                window start, nodes whose data reached the window)
* ``truth``  -- GPS ground truth on ``**/gps/truth`` (data-clock timestamp)
* ``energy`` -- one per node, modality and second that reached the edge (``**/energy``)
* ``event``  -- link changes and schedule arming published by topo.py on ``emu/events``

``analyze.py`` turns a record file into localization error per fault phase.

Runs as an edge service (see docker/emu-edge.yml)::

    python /emu/recorder.py [--out DIR]
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import click
from acies.core import AciesApp, AciesContext, setup_logging
from acies.core.msg import AciesInference

logger = logging.getLogger(__name__)

app = AciesApp()
_lock = threading.Lock()


def _write(ctx: AciesContext, record: dict[str, Any]) -> None:
    line = json.dumps({'wall_ns': time.time_ns(), **record}, default=str)
    with _lock:
        ctx.app['file'].write(line + '\n')
        ctx.app['file'].flush()


@app.on_startup
def setup(ctx: AciesContext) -> None:
    out_dir = Path(ctx.cfg['out_dir']).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{time.strftime("%Y%m%d-%H%M%S", time.gmtime())}.jsonl'
    ctx.app['file'] = path.open('a', buffering=1)
    logger.info('recording to %s', path)


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    ctx.app['file'].close()


@app.subscribe('**/spar')  # spar publishes on joint/spar/spar (<namespace>/<name>/spar)
def on_spar(ctx: AciesContext, msg: AciesInference) -> None:
    for pred in msg.predictions:
        _write(
            ctx,
            {
                'type': 'spar',
                'source': msg.source,
                'label': pred.label,
                'score': pred.score,
                'lat': pred.latitude,
                'lon': pred.longitude,
                'window_start_s': pred.extras.get('window_start_s'),
                'nodes': pred.extras.get('nodes', []),
                'infer_ms': pred.extras.get('infer_ms'),
            },
        )


@app.subscribe('**/gps/truth')
def on_truth(ctx: AciesContext, msg: dict[str, Any]) -> None:
    data_ns = msg.get('timestamp')
    for label, pos in msg.items():
        if label == 'timestamp' or not isinstance(pos, dict):
            continue
        _write(ctx, {'type': 'truth', 'label': label, 'data_ns': data_ns, 'lat': pos['lat'], 'lon': pos['lon']})


@app.subscribe('**/energy')
def on_energy(ctx: AciesContext, msg: dict[str, Any]) -> None:
    """One small message per node, modality and 1-second window: shows which nodes' data reached the edge."""
    node, _, modality = str(msg.get('source', '')).partition('/')
    _write(ctx, {'type': 'energy', 'node': node, 'modality': modality, 'data_ns': msg.get('timestamp')})


@app.subscribe('emu/events')
def on_event(ctx: AciesContext, msg: dict[str, Any]) -> None:
    _write(ctx, {'type': 'event', **msg})


@app.cli()
@click.option('--out', 'out_dir', default='~/.acies/emu/records', show_default=True, help='Output directory.')
def main(out_dir: str) -> None:
    app.state.config.update({'out_dir': out_dir})
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
