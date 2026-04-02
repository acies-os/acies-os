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
from typing import Any

import click
import tomli as tomllib
from acies.corev2 import AciesApp, AciesContext, setup_logging
from acies.corev2.msg import AciesInference

logger = logging.getLogger(__name__)

app = AciesApp()


@app.subscribe('**/vehicle')
def on_vehicle(_ctx: AciesContext, msg: AciesInference) -> None:
    for pred in msg.predictions:
        logger.debug('from %s: %s', msg.source, pred)


@app.subscribe('ws://ctl')
def on_ctl(_ctx: AciesContext, msg: Any) -> None:
    logger.info('ctl command received: %r', msg)


@app.on_startup
def setup(ctx: AciesContext) -> None:
    with open(ctx.cfg['config_path'], 'rb') as f:
        ctx.cfg.update(tomllib.load(f))
    gps: dict[str, list[float]] = ctx.cfg.get('gps', {})
    confidence_threshold: dict[str, dict[str, float]] = ctx.cfg.get('confidence_threshold', {})
    ctx.app['gps'] = gps
    ctx.app['confidence_threshold'] = confidence_threshold
    logger.info('gateway ready: %d node(s) in gps table', len(gps))


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
def main(**kwargs: dict[str, Any]) -> None:
    app.state.config.update(kwargs)
    setup_logging(app.name)
    app.run()


if __name__ == '__main__':
    main()
