# Network Configuration

Most "it builds but nothing shows up in the UI" problems come from addressing. This page lists
every address and port in the system, which `.env` variable controls each one, and how to check
them.

## 1. Topology

The edge server and all Pis must share one IP network, or be routable to each other. Common
setups:

* **Field.** A travel router or the laptop's hotspot provides Wi-Fi. The Pis are flashed to join
  that SSID, and the laptop joins it too.
* **Lab.** All devices sit on the same building or lab LAN.
* **Multi-subnet.** Pis sit behind extra routers (for example `10.7.0.0/24` and `10.7.1.0/24`).
  Add static routes on the edge server. The `add-routes` recipe in the root `justfile` shows
  the pattern:

  ```bash
  sudo route -n add -net 10.7.0.0/24 192.168.68.77    # macOS syntax
  # Linux: sudo ip route add 10.7.0.0/24 via 192.168.68.77
  ```

## 2. Ports

| Port | Proto | Host | Bound by | Controlled by | Who connects |
| --- | --- | --- | --- | --- | --- |
| **7500** | TCP | edge, all interfaces | edge `zenohd` | `ZENOH_LISTEN` (edge `.env`) | Each Pi's `zenohd` |
| 7501 | TCP | every machine, loopback only | local `zenohd` | `ZENOH_CONNECT` | Local services (gateway, sensors, classifiers) |
| **8765** | TCP (WebSocket) | edge | `acies-gateway` | `ACIES_LISTEN` | Browser running the UI |
| **5173** | TCP (HTTP) | edge | `ui` container | `UI_PORT` | Browser |
| **5100** | TCP (HTTP) | edge | `acies-registry` | `REGISTRY_PORT` / `REGISTRY_EXTERNAL` | Each Pi's Docker daemon (image pulls) |
| 1883 | TCP | MQTT broker | external broker | `MQTT_BROKER`, `MQTT_PORT` | `mqtt-gps` (live GPS, field only) |
| 22 | TCP | each Pi | `sshd` | `~/.ssh/config` | Edge server (deploy recipes) |

Ports in **bold** must be reachable over the network. The rest are local only.

:::{note}
`docker/edge.yml` defaults `ZENOH_LISTEN` to `tcp/0.0.0.0:7447`, the Zenoh standard port,
while `.env.edge.example` sets `7500`. Either works, as long as the Pis' `ZENOH_BRIDGE` uses
the **same** port as the edge's `ZENOH_LISTEN`.
:::

## 3. The values you must set

Only a few variables contain an IP address. Fill them in with this table:

| Variable | File | Value | Example |
| --- | --- | --- | --- |
| `ZENOH_LISTEN` | edge `.env` | `tcp/0.0.0.0:<port>` (all interfaces) | `tcp/0.0.0.0:7500` |
| `ZENOH_BRIDGE` | **each Pi** `.env` | `tcp/<EDGE_LAN_IP>:<same port>` | `tcp/192.168.68.50:7500` |
| `REGISTRY_EXTERNAL` | edge `.env` | `<EDGE_LAN_IP>:5100` | `192.168.68.50:5100` |
| `ACIES_LISTEN` | edge `.env` | `ws://127.0.0.1:8765` if you only open the UI on the edge itself; `ws://<EDGE_LAN_IP>:8765` to open it from other devices | `ws://192.168.68.50:8765` |
| `MQTT_BROKER` | edge `.env` | IP of the MQTT broker (field GPS only) | `192.168.70.51` |
| `HostName` per Pi | edge `~/.ssh/config` | `rsN.local` or the Pi's fixed IP | `192.168.68.61` |

`ACIES_LISTEN` does two jobs. The gateway binds to that address, and the UI container passes it
to the browser as `VITE_WS_URL=<ACIES_LISTEN>/ws`. The browser opens exactly that URL, so it must
be an address the **browser's** machine can reach. Neither `127.0.0.1` (from another device) nor
`0.0.0.0` will work there.

## 4. Find the edge server's LAN IP

```bash
# macOS
ipconfig getifaddr en0          # Wi-Fi; try en1 etc. for other interfaces

# Linux
ip -4 addr show | grep inet
hostname -I
```

Pick the address on the interface that shares a network with the Pis. Hotspot interfaces often
use their own subnet (for example `192.168.137.1` on Windows, `10.42.0.1` on Linux
NetworkManager). Use that one, not your upstream Wi-Fi or VPN address.

(static-addresses)=

## 5. Static addresses (strongly recommended)

If the edge server's IP changes, every Pi's `ZENOH_BRIDGE` and `REGISTRY_EXTERNAL` become wrong.
To prevent this:

* Give the edge server and each Pi a **DHCP reservation** on the router, or
* set a static IP on the edge's hotspot or Ethernet interface.

`.local` mDNS names work for SSH but sometimes fail for a few seconds (`scripts/update-pis.sh`
retries for this reason). Use IP addresses inside `.env` files, not `.local` names.

To change the bridge address on all Pis at once after an IP change:

```bash
NEW=tcp/192.168.68.50:7500
for h in $PI_HOSTS; do
  ssh "$h" "cd /ws/acies/acies-os && sed -i 's|^ZENOH_BRIDGE=.*|ZENOH_BRIDGE=$NEW|' .env && grep ZENOH_BRIDGE .env"
done
```

## 6. Firewalls

* **macOS.** *System Settings → Network → Firewall*. Allow incoming connections for Docker,
  Colima, or `zenohd`, or turn the firewall off on a closed field network.
* **Linux (ufw).**
  `sudo ufw allow 7500/tcp && sudo ufw allow 8765/tcp && sudo ufw allow 5173/tcp && sudo ufw allow 5100/tcp`
* **Pis.** Raspberry Pi OS has no firewall by default.
* **Guest or enterprise Wi-Fi** often blocks device-to-device traffic (client isolation). Use
  your own router or hotspot.

## 7. Verify connectivity, step by step

Run the checks in order. Each one depends on the previous.

**a. Pi → edge, IP layer** (on the Pi):

```bash
ping -c 3 <EDGE_LAN_IP>
```

**b. Pi → edge registry:**

```bash
curl -s http://<EDGE_LAN_IP>:5100/v2/_catalog       # {"repositories":["acies-pi"]}
```

**c. Pi → edge Zenoh port** (edge infra must be running, e.g. `just edge-up-infra`):

```bash
nc -vz <EDGE_LAN_IP> 7500                           # "succeeded" / "open"
# or, without nc: timeout 3 bash -c '</dev/tcp/<EDGE_LAN_IP>/7500' && echo open
```

**d. Edge: is zenohd listening on the LAN?**

```bash
lsof -nP -iTCP:7500 -sTCP:LISTEN     # macOS
ss -ltnp | grep 7500                  # Linux
```

**e. Pi zenohd is bridged.** Start the Pi services and check the Pi's `zenohd` output for a
session to the edge address. Then, on the edge with `ACIES_GATEWAY_DEBUG=1` and
`ACIES_LOG=acies=DEBUG`, look for energy readings arriving from that node:

```bash
tail -F ~/.acies/logs/*/infra/gateway.log | grep 'energy from rs1/'
```

**f. Browser → gateway.** Open the UI, then open the browser developer console. A WebSocket
error to `ws://…:8765/ws` means `ACIES_LISTEN` is not reachable from the browser (see section 3).

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `deploy-pi` fails: `http: server gave HTTP response to HTTPS client` | Registry not trusted on that Pi | `just registry-trust <host>` |
| `deploy-pi` fails: `connection refused` to an unexpected IP | `REGISTRY_EXTERNAL` not set; hard-coded default used | Set `REGISTRY_EXTERNAL=<EDGE_LAN_IP>:5100` in edge `.env` |
| Pi containers run but node never appears in UI | Wrong `ZENOH_BRIDGE`, firewall, or port mismatch | Checks 7a–7e |
| Node appears but not on map | `ACIES_NAMESPACE` not in `[gps]` of the gateway's site config | Add the node to `packages/controllers/data/<site>.toml` or rename the node |
| Two nodes overwrite each other | Same `ACIES_NAMESPACE` on two Pis | Make every namespace unique |
| UI loads but stays empty | Browser can't reach `ACIES_LISTEN` | Set `ACIES_LISTEN=ws://<EDGE_LAN_IP>:8765`, restart infra |
| `ssh rs1` hangs or fails intermittently | mDNS flakiness | Use fixed IPs in `~/.ssh/config` |
| Live GPS missing in field | Broker unreachable or wrong topic | `nc -vz $MQTT_BROKER 1883`; check `MQTT_TOPIC` |

## Remote viewing through a jump host

If the UI must be viewed from a machine that can't reach the edge server directly, the root
`justfile` has recipes that tunnel the UI (5173) and gateway (8765) through an SSH host (`eugene`
in the reference setup):

```bash
just serve-via-eugene     # on the edge server: reverse-forward 5173/8765 to the jump host
just receive-via-eugene   # on the viewer: forward local 5173/8765 from the jump host
```

Replace `eugene` with your own SSH host alias. With this setup, leave `ACIES_LISTEN` at
`ws://127.0.0.1:8765`.
