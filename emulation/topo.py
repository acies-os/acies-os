"""Mininet runner for acies-os network emulation.

Builds a star topology inside the ``acies-mn`` container::

    rs1 ─┐                         (each link: TCLink, shaped by tc/netem)
    rs2 ─┤
     …   ├── s1 (Linux bridge) ── edge (root namespace of the container, 10.0.0.254)
    rs10 ┘

Each node host is a network namespace that runs what a Pi runs in
``docker/pi-replay.yml``: its own ``zenohd`` (bridged to the edge zenohd at
10.0.0.254:7500), geo + mic replay, and optionally the split VibroFM encoder +
vehicle head. The edge services (zenohd, gateway, gps, spar, recorder) run from
docker compose in the container's root namespace (``docker/emu-edge.yml``).

Faults from the experiment file are timed relative to the replay start: the
runner watches ``**/ctl/notify/start_at`` and re-arms the schedule on every
deploy. Faults can also be sent live on the ``emu/ctl`` topic (see fault.py).
Every applied change is published on ``emu/events`` for the recorder.

Usage (inside the container; see ``just emu-up``)::

    python /emu/topo.py /emu/experiments/outage.toml [--cli]
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import msgspec
import tomllib
import zenoh
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import Host
from mininet.nodelib import LinuxBridge

logging.setLoggerClass(logging.Logger)  # importing mininet.log installs its own (quiet) logger class
logger = logging.getLogger('emu')

EDGE_NAME = 'edge'
EDGE_IP = '10.0.0.254'
SUBNET = '10.0.0.0/24'
EDGE_ZENOH = f'tcp/{EDGE_IP}:7500'
LOCAL_ZENOH = 'tcp/127.0.0.1:7501'  # inside every namespace (nodes and edge)
# Replays start idle: they wait for this far-future start_at, and a deploy replaces it.
IDLE_START_AT = 4102444800.0  # 2100-01-01
LOG_DIR = Path('/root/.acies/emu/logs')
DEFAULT_NODES = ['rs1', 'rs2', 'rs3', 'rs5', 'rs6', 'rs7', 'rs8', 'rs10']
LINK_KEYS = ('bw', 'delay', 'jitter', 'loss', 'max_queue_size')
ACTIONS = ('down', 'up', 'set', 'reset', 'unplug', 'plug')


# ------------------------------- configuration -------------------------------


def _link_params(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only TCLink parameters; drop unset ones."""
    unknown = set(raw) - set(LINK_KEYS)
    if unknown:
        raise ValueError(f'unknown link parameter(s) {sorted(unknown)}; allowed: {LINK_KEYS}')
    return {k: v for k, v in raw.items() if v is not None}


def load_experiment(path: str) -> dict[str, Any]:
    with open(path, 'rb') as f:
        exp = tomllib.load(f)
    nodes_cfg = exp.setdefault('nodes', {})
    nodes_cfg.setdefault('names', DEFAULT_NODES)
    nodes_cfg.setdefault('scene', os.environ.get('SCENE', '2024-03-29-ICT'))
    nodes_cfg.setdefault('run', int(os.environ.get('RUN', '2')))
    nodes_cfg.setdefault('classifier', True)

    links = exp.setdefault('links', {})
    default = _link_params(links.get('default', {}))
    base: dict[str, dict[str, Any]] = {}
    for name in nodes_cfg['names']:
        base[name] = {**default, **_link_params(links.get(name, {}))}
    base[EDGE_NAME] = _link_params(links.get(EDGE_NAME, {}))  # edge uplink: unshaped unless set
    exp['_base_links'] = base

    events = exp.setdefault('events', [])
    for i, ev in enumerate(events):
        if ev.get('action') not in ACTIONS:
            raise ValueError(f'events[{i}]: action must be one of {ACTIONS}, got {ev.get("action")!r}')
        if 'at' not in ev:
            raise ValueError(f'events[{i}]: missing `at` (seconds after replay start)')
    events.sort(key=lambda ev: float(ev['at']))
    return exp


# --------------------------------- emulator ----------------------------------


class Emulator:
    """Owns the Mininet network, node processes, link state and fault schedule."""

    def __init__(self, exp: dict[str, Any]) -> None:
        self.exp = exp
        self.names: list[str] = list(exp['nodes']['names'])
        self.base: dict[str, dict[str, Any]] = exp['_base_links']
        self.current: dict[str, dict[str, Any]] = {n: dict(p) for n, p in self.base.items()}
        self.down: dict[str, bool] = {n: False for n in self.base}
        self.lock = threading.Lock()  # serializes all Mininet / tc calls
        self.procs: list[tuple[str, subprocess.Popen[bytes]]] = []
        self.session: zenoh.Session | None = None
        self.schedule_gen = 0  # bumped on every re-arm; stale schedule threads exit
        self.armed_start_at: float | None = None
        self.stop = threading.Event()

        self.net = Mininet(
            switch=LinuxBridge,
            controller=None,
            link=TCLink,
            ipBase=SUBNET,
            autoSetMacs=True,
            waitConnected=False,
        )

    # ------------------------------ topology ------------------------------

    def build(self) -> None:
        s1 = self.net.addSwitch('s1')
        edge = self.net.addHost(EDGE_NAME, ip=f'{EDGE_IP}/24', inNamespace=False)
        self.net.addLink(edge, s1, **self.base[EDGE_NAME])
        for i, name in enumerate(self.names, start=1):
            host = self.net.addHost(name, ip=f'10.0.0.{i}/24')
            self.net.addLink(host, s1, **self.base[name])
        self.net.start()
        for name in [*self.names, EDGE_NAME]:
            logger.info('link %-5s %s', name, _fmt_params(self.base[name]))

    def _link(self, name: str) -> Any:
        host = self.net.get(name)
        return self.net.linksBetween(host, self.net.get('s1'))[0]

    # ---------------------------- node services ---------------------------

    def start_nodes(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        nodes = self.exp['nodes']
        for name in self.names:
            host = self.net.get(name)
            self._spawn(
                host,
                'zenohd',
                ['zenohd', '--listen', LOCAL_ZENOH, '--connect', EDGE_ZENOH, '--no-multicast-scouting'],
            )
        time.sleep(1.0)  # let each node's zenohd bind before its clients connect
        for name in self.names:
            host = self.net.get(name)
            for modality in ('geo', 'mic'):
                # --loop: after a playback the replay waits for the next deploy instead of exiting.
                # --start-at far future: idle until the gateway's deploy sets start_at.
                cmd = [
                    'acies-replay', '--data-dir', '/data', '--scene', str(nodes['scene']),
                    '--node', name, '--run', str(nodes['run']), '--modality', modality,
                    '--loop', '--start-at', str(IDLE_START_AT),
                ]  # fmt: skip
                self._spawn(host, modality, cmd)
            if nodes['classifier']:
                self._spawn(
                    host,
                    'vfm_encoder',
                    [
                        'acies-vfm-encoder', '--weight', f'/model/{_env("VFM_ENCODER_WEIGHT")}',
                        '--geo', f'{name}/geo', '--mic', f'{name}/mic', '--drone-mic', f'{name}/drone_mic',
                        '--deactivated',
                    ],
                )  # fmt: skip
                self._spawn(
                    host,
                    'vfm_vehicle',
                    [
                        'acies-vfm-head', '--weight', f'/model/{_env("VFM_VEHICLE_HEAD_WEIGHT")}',
                        '--input', f'{name}/vfm_encoder/vehicle/embedding',
                        '--embed-dim', os.environ.get('VFM_VEHICLE_EMBED_DIM', '512'),
                        '--labels', _env('VFM_LABELS'), '--deactivated',
                    ],
                )  # fmt: skip
        logger.info('started %d process(es) on %d node(s); logs in %s', len(self.procs), len(self.names), LOG_DIR)

    def _spawn(self, host: Host, service: str, cmd: list[str]) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith('ACIES_')}
        env.update(
            ACIES_NAMESPACE=host.name,
            ACIES_NAME=service,
            ACIES_CONNECT=LOCAL_ZENOH,
            ACIES_LOG=os.environ.get('ACIES_LOG', ''),
        )
        log = open(LOG_DIR / f'{host.name}-{service}.log', 'ab')  # noqa: SIM115 -- kept open for the process lifetime
        proc = host.popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.procs.append((f'{host.name}/{service}', proc))

    def watch_procs(self) -> None:
        """Log node processes that exit (they are not restarted)."""
        reported: set[str] = set()
        while not self.stop.wait(5.0):
            for label, proc in self.procs:
                code = proc.poll()
                if code is not None and label not in reported:
                    reported.add(label)
                    logger.error('process %s exited with code %s; see %s', label, code, LOG_DIR)

    # ----------------------------- link control ----------------------------

    def _apply(self, name: str) -> None:
        """Push the effective parameters of one link to both of its interfaces."""
        params = dict(self.current[name])
        if self.down[name]:
            params['loss'] = 100
        link = self._link(name)
        for intf in (link.intf1, link.intf2):
            intf.config(**params)

    def apply_event(self, ev: dict[str, Any], source: str) -> None:
        action: str = ev['action']
        targets = _expand_targets(ev.get('nodes', ev.get('node')), self.names)
        params = _link_params({k: v for k, v in ev.items() if k in LINK_KEYS})
        unknown = [t for t in targets if t not in self.base]
        if unknown:
            logger.error('event %s: unknown node(s) %s; ignoring', ev, unknown)
            return
        with self.lock:
            for name in targets:
                match action:
                    case 'down':
                        self.down[name] = True
                        self._apply(name)
                    case 'up':
                        self.down[name] = False
                        self._apply(name)
                    case 'set':
                        self.current[name].update(params)
                        self._apply(name)
                    case 'reset':
                        self.current[name] = dict(self.base[name])
                        self.down[name] = False
                        self._apply(name)
                    case 'unplug':
                        self.net.configLinkStatus(name, 's1', 'down')
                    case 'plug':
                        self.net.configLinkStatus(name, 's1', 'up')
                        self._apply(name)  # restore shaping on the re-enabled interfaces
        effective = {n: {**self.current[n], **({'loss': 100} if self.down[n] else {})} for n in targets}
        logger.info('[%s] %s %s -> %s', source, action, ','.join(targets), effective)
        self._publish(
            'emu/events',
            {
                'source': source,
                'wall_ns': time.time_ns(),
                'start_at': self.armed_start_at,
                'at': ev.get('at'),
                'action': action,
                'nodes': targets,
                'params': params,
                'effective': effective,
            },
        )

    # ------------------------------ scheduling -----------------------------

    def arm(self, start_at: float) -> None:
        """(Re)start the fault schedule relative to a replay start_at."""
        if start_at >= IDLE_START_AT or start_at == self.armed_start_at:
            return
        self.armed_start_at = start_at
        self.schedule_gen += 1
        gen = self.schedule_gen
        events: list[dict[str, Any]] = self.exp['events']
        logger.info(
            'replay start_at=%.3f (in %.1fs); arming %d event(s)', start_at, start_at - time.time(), len(events)
        )

        # a new deploy starts from the configured baseline
        self.apply_event({'action': 'reset', 'nodes': 'all+edge'}, source='arm')
        self._publish('emu/events', {'source': 'arm', 'wall_ns': time.time_ns(), 'start_at': start_at,
                                     'action': 'armed', 'nodes': [], 'events': events})  # fmt: skip

        def run() -> None:
            for ev in events:
                delay = start_at + float(ev['at']) - time.time()
                if delay > 0 and self.stop.wait(delay):
                    return
                if gen != self.schedule_gen:
                    return  # superseded by a newer deploy
                self.apply_event(ev, source='schedule')

        threading.Thread(target=run, name=f'schedule-{gen}', daemon=True).start()

    # -------------------------------- zenoh --------------------------------

    def connect_zenoh(self) -> None:
        """Connect to the edge zenohd (started later by compose); retry until it is up."""
        cfg = zenoh.Config()
        cfg.insert_json5('mode', '"client"')
        cfg.insert_json5('connect/endpoints', f'["{LOCAL_ZENOH}"]')
        while not self.stop.is_set():
            try:
                session = zenoh.open(cfg)
                break
            except Exception as e:  # noqa: BLE001 -- zenoh raises a generic error when the router is down
                logger.info('waiting for edge zenohd at %s (%s)', LOCAL_ZENOH, e)
                if self.stop.wait(3.0):
                    return
        else:
            return
        self.session = session
        session.declare_subscriber('**/ctl/notify/start_at', self._on_start_at)
        session.declare_subscriber('emu/ctl', self._on_ctl)
        logger.info('connected to edge zenohd; listening for deploys and emu/ctl')

    def _on_start_at(self, sample: zenoh.Sample) -> None:
        try:
            change = msgspec.msgpack.decode(bytes(sample.payload))
            value = change.get('value')
            if change.get('op') == 'set' and isinstance(value, (int, float)):
                self.arm(float(value))
        except Exception:
            logger.exception('bad start_at notification on %s', sample.key_expr)

    def _on_ctl(self, sample: zenoh.Sample) -> None:
        try:
            ev = msgspec.msgpack.decode(bytes(sample.payload))
            if ev.get('action') not in ACTIONS:
                logger.error('emu/ctl: action must be one of %s: %r', ACTIONS, ev)
                return
            self.apply_event(ev, source='ctl')
        except Exception:
            logger.exception('bad emu/ctl message')

    def _publish(self, topic: str, msg: dict[str, Any]) -> None:
        if self.session is not None:
            self.session.put(topic, msgspec.msgpack.encode(msg))

    # ------------------------------- shutdown ------------------------------

    def shutdown(self) -> None:
        self.stop.set()
        for _, proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 10
        for label, proc in self.procs:
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                logger.warning('killing %s', label)
                proc.kill()
        if self.session is not None:
            self.session.close()
        self.net.stop()


# ---------------------------------- helpers ----------------------------------


def _env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        sys.exit(f'environment variable {key} is required (set it in .env)')
    return value


def _expand_targets(value: Any, names: list[str]) -> list[str]:
    """'rs3' | ['rs3', 'rs5'] | 'all' (every node) | 'all+edge' -> list of link names."""
    if value is None:
        raise ValueError('event needs `nodes`')
    if value == 'all':
        return list(names)
    if value == 'all+edge':
        return [*names, EDGE_NAME]
    return [value] if isinstance(value, str) else list(value)


def _fmt_params(params: dict[str, Any]) -> str:
    return ' '.join(f'{k}={v}' for k, v in params.items()) or 'unshaped'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', help='experiment TOML (nodes, links, events)')
    parser.add_argument('--cli', action='store_true', help='open the Mininet CLI (needs `docker run -it`)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    setLogLevel('warning')
    exp = load_experiment(args.experiment)
    logger.info(
        'experiment %s: %d node(s), %d event(s)', args.experiment, len(exp['nodes']['names']), len(exp['events'])
    )

    emu = Emulator(exp)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    signal.signal(signal.SIGINT, lambda *_: done.set())
    try:
        emu.build()
        emu.start_nodes()
        threading.Thread(target=emu.watch_procs, name='watch', daemon=True).start()
        threading.Thread(target=emu.connect_zenoh, name='zenoh', daemon=True).start()
        if args.cli:
            from mininet.cli import CLI

            CLI(emu.net)
        else:
            done.wait()
    finally:
        logger.info('shutting down')
        emu.shutdown()


if __name__ == '__main__':
    main()
