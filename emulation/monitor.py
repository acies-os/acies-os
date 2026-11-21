"""Live network monitor for the emulator: link state, data delivery and SPAR error, on one web page.

Listens on the edge zenohd like any edge service and combines:

* ``emu/state``    -- every link's settings and tc counters (topo.py, once a second)
* ``emu/events``   -- applied link changes and deploys (topo.py)
* ``**/energy``    -- one message per node, modality and 1-s window that reached the edge
* ``**/spar``      -- SPAR localizations
* ``**/gps/truth`` -- GPS ground truth

Serves the web app (``emulation/ui``) and a WebSocket on ``--port`` (``just emu-up-nodes`` publishes it on
http://127.0.0.1:8780). Once a second it sends every browser a JSON snapshot and publishes
the same metrics (without the per-window history) as msgpack on ``emu/metrics``.
The page can change links (forwarded to ``emu/ctl``, like fault.py) and deploy a replay
(like deploy.py).

Clocks: the replays map the run's data origin to ``start_at`` (``acies-replay --origin run``),
so "replay time" is ``data time - origin`` and fault times (``at``) use the same clock. For
every arrival, ``data - wall`` is ``origin - start_at`` minus that message's delay, so its
maximum over all arrivals since the deploy (the least-delayed message) gives the origin as
``start_at + max(data - wall)``. This works when the monitor starts in the middle of a run.
A message's extra delay is that maximum minus its own ``data - wall``: the delay on top of
the fastest arrival, not an absolute one-way delay.

Usage (edge service, see docker/emu-edge.yml)::

    python /emu/monitor.py [--port 8780] [--config-dir /config] [--map ict]

Sites: like the gateway, it loads every site config in ``--config-dir`` and shows the one the
gateway is using (the ``map`` of the last deploy, read from the gateway's ``health`` messages),
or ``--map`` before the first deploy.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import msgspec
import tomllib
import zenoh
from acies.core.msg import AciesInference
from deploy import deploy
from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response
from websockets.sync.client import connect

logger = logging.getLogger('monitor')

_NS = 1_000_000_000
WINDOW_S = 2  # SPAR window length
DELIVERY_WINDOW_S = 10.0  # delivery and delay are computed over the last 10 s
SUMMARY_WINDOW_S = 30.0  # SPAR error summary over the last 30 s of replay time
TRUTH_TRAIL_S = 30
TRUTH_DELAY_S = 2  # the map shows the truth 2 s late, like the main UI, so it lines up with SPAR
MAX_WINDOWS = 1000
EARTH_R = 6_371_000.0
ACTIONS = ('down', 'up', 'set', 'reset', 'unplug', 'plug')
LINK_KEYS = ('bw', 'delay', 'jitter', 'loss', 'max_queue_size')
# The web app (emulation/ui) built into the image by `just emu-build`, or built on the host
# (`npm run build` in emulation/ui, mounted at /emu/ui/dist); the newer build wins.
UI_DIRS = (Path('/opt/emu-ui'), Path(__file__).with_name('ui') / 'dist')
CONTENT_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.json': 'application/json',
    '.woff2': 'font/woff2',
    '.map': 'application/json',
}


def _dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)
    return EARTH_R * math.hypot(x, y)


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


class Monitor:
    """Thread-safe state fed by zenoh callbacks and read by the web server."""

    def __init__(self, config_dir: str, default_map: str, data_dir: str) -> None:
        self.lock = threading.Lock()
        self.sites = _load_sites(config_dir, data_dir)
        self.active_map = default_map if default_map in self.sites else next(iter(self.sites), default_map)
        self.state: dict[str, Any] | None = None  # last emu/state
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self._reset_run(None)

    def _reset_run(self, start_at: float | None) -> None:
        self.start_at = start_at
        self.best_offset_ns: int | None = None  # max(data_ns - wall_ns): the least-delayed arrival
        self.arrivals: deque[tuple[int, str, float]] = deque()  # (wall_ns, node, delay_s)
        self.last_seen: dict[str, int] = {}  # node -> wall_ns
        self.windows: list[dict[str, Any]] = []
        # data second -> label -> (lat, lon); cleared too, or the map would keep drawing the previous
        # replay's truth (the trail ends at the newest data second) until the new one catches up
        self.truth: dict[int, dict[str, tuple[float, float]]] = {}

    # ------------------------------ zenoh input ------------------------------

    def on_state(self, sample: zenoh.Sample) -> None:
        msg = msgspec.msgpack.decode(bytes(sample.payload))
        with self.lock:
            self.state = msg
            if msg.get('start_at') and msg['start_at'] != self.start_at:
                self._reset_run(msg['start_at'])  # missed the "armed" event (monitor started later)

    def on_event(self, sample: zenoh.Sample) -> None:
        msg = msgspec.msgpack.decode(bytes(sample.payload))
        with self.lock:
            if msg.get('action') == 'armed':
                self._reset_run(msg.get('start_at'))
            msg.pop('events', None)
            self.events.append(msg)

    def on_energy(self, sample: zenoh.Sample) -> None:
        wall_ns = time.time_ns()
        msg = msgspec.msgpack.decode(bytes(sample.payload))
        node = str(msg.get('source', '')).partition('/')[0]
        data_ns = msg.get('timestamp')
        if not node or not isinstance(data_ns, int):
            return
        with self.lock:
            if self.start_at is None or wall_ns < self.start_at * _NS:
                return
            offset = data_ns - wall_ns
            if self.best_offset_ns is None or offset > self.best_offset_ns:
                self.best_offset_ns = offset
            self.arrivals.append((wall_ns, node, (self.best_offset_ns - offset) / _NS))
            self.last_seen[node] = wall_ns

    def on_truth(self, sample: zenoh.Sample) -> None:
        msg = msgspec.msgpack.decode(bytes(sample.payload))
        data_ns = msg.get('timestamp')
        if not isinstance(data_ns, int):
            return
        positions = {
            label: (pos['lat'], pos['lon'])
            for label, pos in msg.items()
            if label != 'timestamp' and isinstance(pos, dict) and 'lat' in pos
        }
        with self.lock:
            self.truth[data_ns // _NS] = positions
            if len(self.truth) > 4000:
                for sec in sorted(self.truth)[:1000]:
                    del self.truth[sec]

    def on_spar(self, sample: zenoh.Sample) -> None:
        wall_ns = time.time_ns()
        msg = msgspec.msgpack.decode(bytes(sample.payload), type=AciesInference)
        with self.lock:
            for pred in msg.predictions:
                ws = pred.extras.get('window_start_s')
                if ws is None or pred.latitude is None or pred.longitude is None:
                    continue
                truth = self._truth_near(int(ws), pred.latitude, pred.longitude)
                origin_s = self._origin_s()
                t = int(ws) - origin_s if origin_s is not None else None
                lag = (wall_ns / _NS - self.start_at) - (t + WINDOW_S) if t is not None and self.start_at else None
                self.windows.append(
                    {
                        't': t,
                        'window_start_s': int(ws),
                        'label': pred.label,
                        'score': round(pred.score, 3),
                        'lat': pred.latitude,
                        'lon': pred.longitude,
                        'truth': truth[:2] if truth else None,
                        'error_m': round(truth[2], 1) if truth else None,
                        'nodes': list(pred.extras.get('nodes', [])),
                        'lag_s': round(lag, 2) if lag is not None else None,
                    }
                )
            del self.windows[:-MAX_WINDOWS]

    def _origin_s(self) -> int | None:
        """Data second that played at start_at (see the module docstring)."""
        if self.start_at is None or self.best_offset_ns is None:
            return None
        # the estimate is early by the fastest link's delay (ms); floor() then matches analyze.py
        return (int(self.start_at * _NS) + self.best_offset_ns + _NS // 20) // _NS

    def _truth_near(self, sec: int, lat: float, lon: float) -> tuple[float, float, float] | None:
        """Nearest vehicle's truth at the window start (±1 s): (lat, lon, error in metres)."""
        for d in (0, 1, -1):
            positions = self.truth.get(sec + d)
            if positions:
                best = min(positions.values(), key=lambda p: _dist_m(lat, lon, *p))
                return best[0], best[1], _dist_m(lat, lon, *best)
        return None

    # -------------------------------- output ---------------------------------

    def snapshot(self, *, history: bool = True) -> dict[str, Any]:
        now_ns = time.time_ns()
        with self.lock:
            cutoff = now_ns - int(DELIVERY_WINDOW_S * _NS)
            while self.arrivals and self.arrivals[0][0] < cutoff:
                self.arrivals.popleft()
            state = self.state or {}
            nodes: list[str] = state.get('nodes') or sorted(self.site()['nodes'])
            links: dict[str, Any] = {k: dict(v) for k, v in state.get('links', {}).items()}
            for node in nodes:
                delays = [d for _, n, d in self.arrivals if n == node]
                links.setdefault(node, {}).update(
                    {
                        # geo + mic: 2 messages per second when everything arrives
                        'delivery_pct': round(min(100.0, 100 * len(delays) / (2 * DELIVERY_WINDOW_S))),
                        'delay_ms': round(1000 * statistics.median(delays)) if delays else None,
                        'delay_max_ms': round(1000 * max(delays)) if delays else None,
                        'last_data_s': round((now_ns - self.last_seen[node]) / _NS, 1)
                        if node in self.last_seen
                        else None,
                    }
                )
            t_now = now_ns / _NS - self.start_at if self.start_at else None
            recent = [
                w
                for w in self.windows
                if t_now is not None and w['t'] is not None and w['t'] >= t_now - SUMMARY_WINDOW_S
            ]
            errors = [w['error_m'] for w in recent if w['error_m'] is not None]
            last = self.windows[-1] if self.windows else None
            spar = {
                'windows_total': len(self.windows),
                'windows_30s': len(recent),
                'expected_30s': int(SUMMARY_WINDOW_S // WINDOW_S),
                'error_median_30s': round(statistics.median(errors), 1) if errors else None,
                'error_p90_30s': round(_pct(errors, 0.9), 1) if errors else None,
                'nodes_mean_30s': round(statistics.mean(len(w['nodes']) for w in recent), 1) if recent else None,
                'lag_30s': round(statistics.mean(w['lag_s'] for w in recent if w['lag_s'] is not None), 2)
                if any(w['lag_s'] is not None for w in recent)
                else None,
                'last': last,
            }
            snap: dict[str, Any] = {
                'type': 'snapshot',
                'wall_ns': now_ns,
                'experiment': state.get('experiment'),
                'map': self.active_map,
                'start_at': self.start_at,
                't_replay': round(t_now, 1) if t_now is not None else None,
                'schedule': state.get('events', []),
                'nodes': nodes,
                'links': links,
                'spar': spar,
            }
            if history:
                snap['windows'] = list(self.windows)
                snap['events'] = list(self.events)[-50:]
                snap['truth_trail'] = self._truth_trail()
            return snap

    def site(self) -> dict[str, Any]:
        return self.sites.get(self.active_map, {'nodes': {}, 'road': []})

    def set_map(self, name: str) -> None:
        if name in self.sites and name != self.active_map:
            logger.info('site: %s -> %s', self.active_map, name)
            self.active_map = name

    def _truth_trail(self) -> dict[str, list[tuple[float, float]]]:
        """TRUTH_TRAIL_S seconds of each vehicle's truth (by data time), ending TRUTH_DELAY_S ago."""
        if not self.truth:
            return {}
        end = max(self.truth) - TRUTH_DELAY_S
        trail: dict[str, list[tuple[float, float]]] = {}
        for sec in range(end - TRUTH_TRAIL_S, end + 1):
            for label, pos in self.truth.get(sec, {}).items():
                trail.setdefault(label, []).append(pos)
        return trail


def _load_sites(config_dir: str, data_dir: str) -> dict[str, dict[str, Any]]:
    """Node positions, road and targets of every gateway site config (packages/controllers/data/*.toml).

    A target is a ``[routes]`` entry (``key = "scene/runN"``), as in a deploy. ``nodes_with_data``
    counts the site's nodes whose geo recording for that run is in ``data_dir``, so the page can
    grey out targets this server cannot replay.
    """
    sites: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(config_dir).glob('*.toml')):
        try:
            with path.open('rb') as f:
                cfg = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            logger.warning('skipping site config %s: %s', path, e)
            continue
        nodes = {k: v for k, v in cfg.get('gps', {}).items() if isinstance(v, list)}
        targets = {}
        for key, route in cfg.get('routes', {}).items():
            scene, _, run = str(route).partition('/')
            have = [n for n in nodes if (Path(data_dir) / scene / f'{run}_{n}_geo.parquet').is_file()]
            targets[key] = {'route': route, 'nodes_with_data': len(have)}
        sites[path.stem] = {'nodes': nodes, 'road': cfg.get('road', {}).get('coordinates', []), 'targets': targets}
    if not sites:
        logger.warning('no site configs in %s; the map will show no nodes', config_dir)
    logger.info('site configs: %s', list(sites))
    return sites


def watch_gateway(url: str, monitor: Monitor) -> None:
    """Follow the gateway's active site from its ``health`` messages (it changes on each deploy)."""
    while True:
        try:
            with connect(url) as ws:
                for raw in ws:
                    topic, payload = msgspec.msgpack.decode(raw)
                    if topic in ('health', 'ws://health'):
                        name = msgspec.msgpack.decode(payload).get('map')
                        if name:
                            with monitor.lock:
                                monitor.set_map(name)
        except Exception as e:  # noqa: BLE001 -- the gateway may not be up yet; keep retrying
            logger.debug('gateway watch: %s', e)
        time.sleep(3.0)


# ---------------------------------- zenoh ----------------------------------


def open_zenoh(connect: str, monitor: Monitor) -> zenoh.Session:
    cfg = zenoh.Config()
    cfg.insert_json5('mode', '"client"')
    cfg.insert_json5('connect/endpoints', f'["{connect}"]')
    while True:
        try:
            session = zenoh.open(cfg)
            break
        except Exception as e:  # noqa: BLE001 -- zenoh raises a generic error when the router is down
            logger.info('waiting for zenohd at %s (%s)', connect, e)
            time.sleep(3.0)
    for key, fn in (
        ('emu/state', monitor.on_state),
        ('emu/events', monitor.on_event),
        ('**/energy', monitor.on_energy),
        ('**/spar', monitor.on_spar),
        ('**/gps/truth', monitor.on_truth),
    ):
        session.declare_subscriber(key, _guard(fn))
    logger.info('connected to zenohd at %s', connect)
    return session


def _guard(fn: Any) -> Any:
    def wrapped(sample: zenoh.Sample) -> None:
        try:
            fn(sample)
        except Exception:
            logger.exception('bad message on %s', sample.key_expr)

    return wrapped


# --------------------------------- web side ---------------------------------


def _handle_command(cmd: dict[str, Any], session: zenoh.Session, monitor: Monitor, gateway: str) -> str:
    """Apply one command from the page; returns a short message for the page."""
    match cmd.get('type'):
        case 'fault':
            action = cmd.get('action')
            nodes = cmd.get('nodes')
            if action not in ACTIONS or not nodes:
                raise ValueError(f'bad fault command {cmd!r}')
            ev: dict[str, Any] = {'action': action, 'nodes': nodes}
            for key in LINK_KEYS:
                if cmd.get(key) not in (None, ''):
                    ev[key] = cmd[key]
            if action == 'set' and len(ev) == 2:
                raise ValueError('set needs at least one of bw, delay, jitter, loss')
            session.put('emu/ctl', msgspec.msgpack.encode(ev))
            return f'sent {action} to {nodes}'
        case 'deploy':
            target = str(cmd.get('target') or '').strip()
            if not target:
                raise ValueError('deploy needs a target')
            with monitor.lock:
                nodes = (monitor.state or {}).get('nodes')
                site = cmd.get('map') or monitor.active_map
                if site not in monitor.sites:
                    raise ValueError(f'unknown site {site!r}; known: {list(monitor.sites)}')
            plan = cmd.get('plan')
            if plan is not None:  # the page's fault plan replaces the schedule before the deploy arms it
                session.put('emu/ctl', msgspec.msgpack.encode({'action': 'plan', 'events': _check_plan(plan)}))
                time.sleep(0.3)
            deploy(target, map_name=site, nodes=nodes, gateway=gateway)
            with monitor.lock:
                monitor.set_map(site)
            return f'deployed {target} at {site} with {len(plan or [])} planned change(s)'
        case 'plan':  # replace the running replay's schedule; past times are skipped
            events = _check_plan(cmd.get('plan'))
            session.put('emu/ctl', msgspec.msgpack.encode({'action': 'plan', 'events': events}))
            return f'applied a plan with {len(events)} change(s) to the running replay'
        case _:
            raise ValueError(f'unknown command {cmd!r}')


def _check_plan(plan: Any) -> list[dict[str, Any]]:
    """Validate the page's plan: a list of {at, nodes, action, bw/delay/jitter/loss}."""
    if not isinstance(plan, list):
        raise TypeError('plan must be a list')
    events = []
    for i, ev in enumerate(plan):
        if not isinstance(ev, dict) or ev.get('action') not in ACTIONS or not ev.get('nodes'):
            raise ValueError(f'plan row {i + 1}: needs links and an action')
        at = float(ev.get('at', -1))
        if at < 0:
            raise ValueError(f'plan row {i + 1}: time must be 0 s or later')
        events.append({'at': at, 'nodes': ev['nodes'], 'action': ev['action'],
                       **{k: ev[k] for k in LINK_KEYS if ev.get(k) not in (None, '')}})  # fmt: skip
    return events


def _ui_dir() -> Path | None:
    built = [d for d in UI_DIRS if (d / 'index.html').is_file()]
    return max(built, key=lambda d: (d / 'index.html').stat().st_mtime) if built else None


def _file_response(path: Path, content_type: str) -> Response:
    body = path.read_bytes()
    headers = Headers([('Content-Type', content_type), ('Content-Length', str(len(body)))])
    return Response(200, 'OK', headers, body)


async def serve_web(port: int, monitor: Monitor, session: zenoh.Session, gateway: str) -> None:
    clients: set[ServerConnection] = set()

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        if request.path == '/ws':
            return None
        root = _ui_dir()
        if root is None:
            return connection.respond(
                503, 'The web app is not built: run `just emu-build` (or npm run build in emulation/ui).\n'
            )
        rel = request.path.split('?', 1)[0].lstrip('/') or 'index.html'
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            return connection.respond(404, 'not found\n')
        if not path.is_file():
            if '.' in Path(rel).name:
                return connection.respond(404, 'not found\n')
            path = root / 'index.html'  # app routes fall back to the app
        return _file_response(path, CONTENT_TYPES.get(path.suffix, 'application/octet-stream'))

    async def handler(ws: ServerConnection) -> None:
        clients.add(ws)
        try:
            await ws.send(json.dumps({'type': 'sites', 'sites': monitor.sites}))
            await ws.send(json.dumps(monitor.snapshot()))
            async for raw in ws:
                try:
                    reply = await asyncio.to_thread(_handle_command, json.loads(raw), session, monitor, gateway)
                    await ws.send(json.dumps({'type': 'ack', 'ok': True, 'message': reply}))
                    logger.info('page command: %s', reply)
                except Exception as e:  # noqa: BLE001 -- report any failure back to the page
                    await ws.send(json.dumps({'type': 'ack', 'ok': False, 'message': str(e)}))
        finally:
            clients.discard(ws)

    async def tick() -> None:
        while True:
            await asyncio.sleep(1.0)
            snap = monitor.snapshot()
            if clients:
                broadcast(clients, json.dumps(snap))
            metrics = {k: v for k, v in snap.items() if k not in ('windows', 'events', 'truth_trail')}
            with contextlib.suppress(Exception):
                session.put('emu/metrics', msgspec.msgpack.encode(metrics))

    async with serve(handler, '0.0.0.0', port, process_request=process_request):
        logger.info('monitor page on http://0.0.0.0:%d', port)
        await tick()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--config-dir', default='/config', help='gateway site configs (*.toml)')
    parser.add_argument('--data-dir', default='/data', help='replay data, to tell which targets can play')
    parser.add_argument('--map', default='ict', help='site to show before the first deploy (file stem or .toml name)')
    parser.add_argument('--connect', default='tcp/127.0.0.1:7501', help='edge zenohd endpoint')
    parser.add_argument('--gateway', default='ws://127.0.0.1:8765/ws', help='gateway WebSocket, for deploys')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    logging.getLogger('websockets.server').setLevel(logging.WARNING)  # one INFO line per page load
    monitor = Monitor(args.config_dir, Path(args.map).stem, args.data_dir)
    session = open_zenoh(args.connect, monitor)
    threading.Thread(target=watch_gateway, args=(args.gateway, monitor), name='gateway', daemon=True).start()
    try:
        asyncio.run(serve_web(args.port, monitor, session, args.gateway))
    finally:
        session.close()


if __name__ == '__main__':
    main()
