# 5. Network

The edge server and all devices must be on the same network (same Wi-Fi or LAN) and able to
reach each other. Guest and enterprise Wi-Fi often block device-to-device traffic, so use your
own router or a hotspot.

## Ports

| Port | On | Used by | Set in |
| --- | --- | --- | --- |
| 7500 | edge server | devices send data to the edge server's zenohd | `ZENOH_LISTEN` (edge) / `ZENOH_BRIDGE` (devices) |
| 8765 | edge server | browser ↔ gateway (WebSocket) | `ACIES_LISTEN` (edge) |
| 5173 | edge server | browser loads the UI | `UI_PORT` (edge) |
| 5100 | edge server | devices pull the Docker image | `REGISTRY_EXTERNAL` (edge) |
| 22 | each device | edge server logs in over SSH | `~/.ssh/config` (edge) |

Port 7501 also appears in the `.env` files (`ZENOH_CONNECT`). It's used only inside each
machine, so leave it as is.

If a firewall is on, allow incoming 7500, 8765, 5173, and 5100 on the edge server.

## Addresses to fill in

Find the edge server's IP on the shared network:

```bash
ipconfig getifaddr en0     # macOS (Wi-Fi)
hostname -I                # Linux
```

Then set:

| Variable | File | Value |
| --- | --- | --- |
| `ZENOH_BRIDGE` | each device `.env` | `tcp/<EDGE_IP>:7500` |
| `REGISTRY_EXTERNAL` | edge `.env` | `<EDGE_IP>:5100` |
| `ACIES_LISTEN` | edge `.env` | `ws://<EDGE_IP>:8765` (`ws://127.0.0.1:8765` if the browser runs on the edge server) |
| `HostName` for each device | edge `~/.ssh/config` | the device's IP |

Give the edge server and the devices **fixed IPs** (a DHCP reservation on the router). If the
edge server's IP changes, every device's `ZENOH_BRIDGE` and the registry address become wrong.

## Check connectivity

Run these on a device, in order:

```bash
ping -c 3 <EDGE_IP>                                   # 1. network reachable
curl -s http://<EDGE_IP>:5100/v2/_catalog             # 2. registry: {"repositories":["acies-pi"]}
nc -vz <EDGE_IP> 7500                                 # 3. zenohd port open (edge services must be running)
```

## Common problems

| Symptom | Fix |
| --- | --- |
| `deploy-pi`: *server gave HTTP response to HTTPS client* | Run `just registry-trust <device>` |
| `deploy-pi` connects to an unknown IP | `REGISTRY_EXTERNAL` is missing from the edge `.env` |
| Device runs but never shows up in the UI | Wrong `ZENOH_BRIDGE` or firewall. Run check 3. |
| Device shows up but not on the map | `ACIES_NAMESPACE` is not one of `rs1, rs2, rs3, rs5, rs6, rs7, rs8, rs10` |
| UI page loads but stays empty | The browser can't reach `ACIES_LISTEN`. Use the edge IP, not `127.0.0.1`. |
