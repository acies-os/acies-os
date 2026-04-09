"""MQTT GPS bridge for AciesOS.

Subscribes to an MQTT broker on ``/+/gps`` and forwards each GPS fix
to the Zenoh mesh as a plain dict on ``<namespace>/<name>/truth/gps``.

MQTT message format (JSON):
  {"lt": <latitude>, "ln": <longitude>}

MQTT topic format:
  /<vehicle_id>/gps  (e.g. /ATV2/gps)

Published Zenoh message:
  {"vehicle_id": "<vehicle_id>", "lat": <float>, "lon": <float>}

Usage::

    acies-mqtt-gps --broker 192.168.70.51
                   [--port 1883]
                   [--mqtt-topic /+/gps]
                   [--acies-namespace NS] [--acies-name NAME]
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any

import click
import paho.mqtt.client as mqtt
from acies.core import AciesApp, AciesContext, setup_logging

logger = logging.getLogger(__name__)

_RECONNECT_DELAY_S = 5.0

app = AciesApp()


def _on_connect(client: mqtt.Client, userdata: dict[str, Any], _flags: Any, rc: int) -> None:
    if rc == 0:
        logger.info('MQTT connected; subscribing to %s', userdata['mqtt_topic'])
        _ = client.subscribe(userdata['mqtt_topic'])
    else:
        logger.warning('MQTT connection failed with code %d', rc)


def _on_disconnect(_client: mqtt.Client, _userdata: dict[str, Any], rc: int) -> None:
    if rc != 0:
        logger.warning('MQTT disconnected unexpectedly (rc=%d); will reconnect', rc)


def _on_message(_client: mqtt.Client, userdata: dict[str, Any], msg: mqtt.MQTTMessage) -> None:
    q: queue.Queue[dict[str, Any]] = userdata['queue']
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        # topic is e.g. /ATV2/gps -> vehicle_id = ATV2
        parts = msg.topic.split('/')
        vehicle_id = parts[1] if len(parts) >= 2 else 'unknown'
        q.put_nowait(
            {'vehicle_id': vehicle_id, 'lat': float(payload.get('lt', 0.0)), 'lon': float(payload.get('ln', 0.0))}
        )
    except Exception:
        logger.exception('failed to parse MQTT message on %s: %r', msg.topic, msg.payload)


@app.on_startup
def setup(ctx: AciesContext) -> None:
    broker: str = ctx.cfg['broker']
    port: int = ctx.cfg['port']
    mqtt_topic: str = ctx.cfg['mqtt_topic']

    q: queue.Queue[dict[str, Any]] = queue.Queue()
    userdata: dict[str, Any] = {'queue': q, 'mqtt_topic': mqtt_topic}

    client = mqtt.Client(userdata=userdata)
    client.on_connect = _on_connect
    client.on_disconnect = _on_disconnect
    client.on_message = _on_message

    ctx.app['client'] = client
    ctx.app['queue'] = q
    ctx.app['broker'] = broker
    ctx.app['port'] = port

    try:
        err_code = client.connect(broker, port, keepalive=60)
        logger.debug('MQTT connect attempt returned code %d', err_code)
    except Exception:
        logger.exception('initial MQTT connect to %s:%d failed; will retry in thread', broker, port)

    err_code = client.loop_start()
    logger.debug('MQTT loop_start returned code %d', err_code)
    logger.info('MQTT client started, broker=%s:%d topic=%s', broker, port, mqtt_topic)


@app.on_shutdown
def teardown(ctx: AciesContext) -> None:
    client: mqtt.Client = ctx.app['client']
    err_code = client.loop_stop()
    logger.debug('MQTT loop_stop returned code %d', err_code)
    err_code = client.disconnect()
    logger.debug('MQTT disconnect returned code %d', err_code)
    logger.info('MQTT client stopped')


@app.thread
def publish(ctx: AciesContext, stop: threading.Event) -> None:
    q: queue.Queue[dict[str, Any]] = ctx.app['queue']
    client: mqtt.Client = ctx.app['client']
    broker: str = ctx.app['broker']
    port: int = ctx.app['port']
    topic: str = ctx.ns.topic('truth', 'gps')

    while not stop.is_set():
        # --- reconnect if needed ---
        if not client.is_connected():
            logger.info('MQTT not connected; retrying in %.1fs', _RECONNECT_DELAY_S)
            _ = stop.wait(timeout=_RECONNECT_DELAY_S)
            if stop.is_set():
                return
            try:
                err_code = client.reconnect()
                logger.debug('MQTT reconnect attempt returned code %d', err_code)
            except Exception:
                logger.exception('MQTT reconnect to %s:%d failed', broker, port)
            continue

        # --- drain incoming GPS fixes ---
        try:
            fix = q.get(timeout=1.0)
        except queue.Empty:
            continue

        ctx.publish(topic, fix)
        logger.debug('gps fix: vehicle=%s lat=%.6f lon=%.6f', fix['vehicle_id'], fix['lat'], fix['lon'])


@app.cli()
@click.option('--broker', required=True, help='MQTT broker hostname or IP.')
@click.option('--port', default=1883, type=int, show_default=True, help='MQTT broker port.')
@click.option(
    '--mqtt-topic',
    default='/+/gps',
    show_default=True,
    help='MQTT topic pattern to subscribe to.',
)
def main(broker: str, port: int, mqtt_topic: str) -> None:
    app.state.config.update({'broker': broker, 'port': port, 'mqtt_topic': mqtt_topic})
    setup_logging(app.name, app.namespace)
    app.run()


if __name__ == '__main__':
    main()
