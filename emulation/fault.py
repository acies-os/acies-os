"""Send a live link change to the running emulator (topo.py) over ``emu/ctl``.

Examples (from the host: ``just emu-fault rs3 down``)::

    python /emu/fault.py rs3 down                  # 100% loss both ways (radio blackout)
    python /emu/fault.py rs3 up
    python /emu/fault.py rs3,rs5 set --loss 20 --delay 100ms
    python /emu/fault.py all set --bw 0.2           # Mbit/s
    python /emu/fault.py edge set --delay 300ms     # edge uplink: affects every node
    python /emu/fault.py all reset                  # back to the experiment's link config
    python /emu/fault.py rs3 unplug                 # interface down (connection errors, not silence)
"""

from __future__ import annotations

import argparse
import time

import msgspec
import zenoh

ACTIONS = ('down', 'up', 'set', 'reset', 'unplug', 'plug')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('nodes', help="comma-separated nodes, 'all', 'all+edge', or 'edge'")
    parser.add_argument('action', choices=ACTIONS)
    parser.add_argument('--bw', type=float, help='bandwidth in Mbit/s')
    parser.add_argument('--delay', help="one-way delay per interface, e.g. '100ms' (RTT is twice this)")
    parser.add_argument('--jitter', help="e.g. '20ms'")
    parser.add_argument('--loss', type=float, help='packet loss in percent')
    parser.add_argument('--max-queue-size', type=int, help='queue length in packets')
    parser.add_argument('--connect', default='tcp/127.0.0.1:7501', help='edge zenohd endpoint')
    args = parser.parse_args()

    nodes: str | list[str] = args.nodes if args.nodes in ('all', 'all+edge') else args.nodes.split(',')
    ev: dict[str, object] = {'action': args.action, 'nodes': nodes}
    for key in ('bw', 'delay', 'jitter', 'loss', 'max_queue_size'):
        value = getattr(args, key)
        if value is not None:
            ev[key] = value
    if args.action == 'set' and len(ev) == 2:
        parser.error('set needs at least one of --bw/--delay/--jitter/--loss/--max-queue-size')

    cfg = zenoh.Config()
    cfg.insert_json5('mode', '"client"')
    cfg.insert_json5('connect/endpoints', f'["{args.connect}"]')
    with zenoh.open(cfg) as session:
        session.put('emu/ctl', msgspec.msgpack.encode(ev))
        time.sleep(0.5)  # let the put flush before the session closes
    print(f'sent {ev}')


if __name__ == '__main__':
    main()
