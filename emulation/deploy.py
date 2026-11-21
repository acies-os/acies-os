"""Deploy a replay through the gateway without the UI.

Sends the same ``ctl`` frame the UI's Deploy button sends. In joint mode every
node is set to SPAR, so the nodes stream raw geo + mic to the edge, where SPAR
localizes the vehicle.

Examples (from the host: ``just emu-deploy`` deploys GLE350)::

    python /emu/deploy.py --target gle350                     # joint (SPAR) on all ICT nodes
    python /emu/deploy.py --target gle350 --mode single       # per-node VibroFM (fm pipeline)
"""

from __future__ import annotations

import argparse
from typing import Any

import msgspec
from websockets.sync.client import connect

ICT_NODES = 'rs1,rs2,rs3,rs5,rs6,rs7,rs8,rs10'


def deploy(
    target: str,
    *,
    map_name: str = 'ict',
    mode: str = 'joint',
    nodes: list[str] | None = None,
    gateway: str = 'ws://127.0.0.1:8765/ws',
) -> dict[str, Any]:
    """Send a deploy to the gateway, exactly as the UI's Deploy button does. Returns the payload."""
    entries = []
    for node_id in nodes or ICT_NODES.split(','):
        if mode == 'joint':
            entries.append({'nodeId': node_id, 'model': 'SPAR', 'modality': 'both'})
        else:
            entries.append(
                {
                    'nodeId': node_id,
                    'model': 'VibroFM',
                    'modality': 'both',
                    'pipeline': 'fm',
                    'vehicleModels': ['fm'],
                    'drone': None,
                    'droneModels': [],
                }
            )
    payload = {
        'appType': 'joint-nodes' if mode == 'joint' else 'single-node',
        'map': map_name,
        'target': target.split(','),
        'nodes': entries,
    }
    with connect(gateway) as ws:
        ws.send(msgspec.msgpack.encode(['ctl', msgspec.msgpack.encode(payload)]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default='gle350', help="route key(s) in the map config, e.g. 'gle350' or 'cx30'")
    parser.add_argument('--map', default='ict', help='site config stem (ict, wp, gq)')
    parser.add_argument('--mode', choices=('joint', 'single'), default='joint')
    parser.add_argument('--nodes', default=ICT_NODES, help='comma-separated node ids')
    parser.add_argument('--gateway', default='ws://127.0.0.1:8765/ws')
    args = parser.parse_args()

    payload = deploy(args.target, map_name=args.map, mode=args.mode, nodes=args.nodes.split(','), gateway=args.gateway)
    print(
        f'deployed {payload["appType"]} target={payload["target"]} on {len(payload["nodes"])} node(s);'
        ' replay starts in ~30 s'
    )


if __name__ == '__main__':
    main()
