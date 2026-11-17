# System Overview

Acies-OS is a content-centric middleware for edge AI. Sensor nodes publish data under
hierarchical **topics** (for example `rs1/geo`, `rs1/mic`), classifiers subscribe to those topics
and publish results (for example `rs1/vfm_vehicle/vehicle`), and an edge **gateway** aggregates
everything and streams it to a browser **UI**. All messaging runs over
[Zenoh](https://zenoh.io).

## Components

### Edge server (laptop / workstation)

Runs from the `acies-server` image unless noted.

| Service | Command | Role |
| --- | --- | --- |
| `zenohd` | `zenohd` | Zenoh router. Listens on the LAN for Pis and on loopback for local services. |
| `gateway` | `acies-gateway` | Subscribes to all node topics, applies the site config (`ict.toml`, `wp.toml`, `gq.toml`), serves a WebSocket to the UI, and activates/deactivates models when the UI config panel is deployed. |
| `ui` | `python -m http.server` (`acies-ui` image) | Static Vue/Leaflet web app. Connects to the gateway WebSocket. |
| `tracker` *(optional)* | `acies-tracker` | Particle-filter vehicle tracker using energy readings from the nodes. |
| `gps` *(optional)* | `acies-gps` | Replays a recorded GPS trace (lab/demo). |
| `mqtt-gps` *(optional)* | `acies-mqtt` | Bridges live GPS from an MQTT broker (field). |
| `edge-spar` *(optional)* | `acies-spar` | Joint (multi-node) SPAR classifier. |

### Raspberry Pi sensor node

Runs from the `acies-pi` image.

| Service | Command | Role |
| --- | --- | --- |
| `zenohd` | `zenohd` | Local Zenoh router on `127.0.0.1:7501`; bridges to the edge server. |
| `geo` | `acies-geo` *(live)* or `acies-replay --modality geo` | Seismic (geophone) data from the Raspberry Shake board over `/dev/serial0`. |
| `mic` | `acies-mic` *(live)* or `acies-replay --modality mic` | Audio from the USB microphone array. |
| `drone_mic` | `acies-replay --modality drone_mic` | Replayed drone audio (demo only). Starts deactivated. |
| `e2e_vehicle` | `acies-vfm` | End-to-end VibroFM vehicle classifier. |
| `vfm_encoder` | `acies-vfm-encoder` | Shared VibroFM encoder (split "fm" pipeline). |
| `vfm_vehicle`, `vfm_drone` | `acies-vfm-head` | Task heads on top of the shared encoder. |
| `e2e_drone` | `acies-vfm --modality audio` | End-to-end drone classifier (AuDroK). |

Every classifier starts **deactivated**; the gateway turns on the pipeline chosen in the UI
("fm" or "e2e", with or without drone).

## Data flow

```text
 Raspberry Pi (rs1)                                   Edge server
 ┌───────────────────────────────┐                   ┌─────────────────────────────────────┐
 │ geo ─┐                        │                   │                                     │
 │ mic ─┼─► zenohd 127.0.0.1:7501├── tcp ──────────► │ zenohd 0.0.0.0:7500 (ZENOH_LISTEN)  │
 │ vfm ─┘    (--connect EDGE)    │   LAN / Wi-Fi     │        127.0.0.1:7501 (ZENOH_CONNECT)│
 └───────────────────────────────┘                   │            │                        │
 Raspberry Pi (rs2) ... same ───────────────────────►│            ▼                        │
                                                     │ gateway ── ws://…:8765/ws ──► UI    │
                                                     │ tracker, gps, spar                  │
                                                     │                      (browser :5173)│
                                                     └─────────────────────────────────────┘
```

Each process connects only to the **local** `zenohd` on `127.0.0.1:7501`. Only the two `zenohd`
routers talk across the network, so the one address you have to get right is the edge server's
LAN IP in each Pi's `ZENOH_BRIDGE`. See [Network configuration](05-network.md).

## Topics and namespaces

* Each Pi has a **namespace** equal to its node name (`ACIES_NAMESPACE=rs1`). Its topics are
  `rs1/geo`, `rs1/mic`, `rs1/vfm_encoder/vehicle/embedding`, `rs1/vfm_vehicle/vehicle`, …
* Edge infrastructure uses `<edge-hostname>/infra` (gateway, tracker, gps).
* The joint SPAR model uses the fixed namespace `joint`.
* Node names must match the keys in the gateway's site config, for example the `[gps]` table in
  `packages/controllers/data/ict.toml` (`rs1 = [lat, lon]`). A node that is not listed there
  will not be placed on the map.

## Deployment scenarios

The same images support three setups. Pick one before you start.

| Scenario | Where sensors run | Data source | Use it for |
| --- | --- | --- | --- |
| **1. Field** | Pis | Live geophone + microphone | Real experiments |
| **2. Lab (edge-only)** | Edge server (simulated nodes) | Replay parquet files on the edge | Development without any Pi |
| **3. Tabletop demo** | Pis | Replay parquet files on each Pi | Demos with real Pi hardware and networking |

Scenario 2 needs only the edge server, so it is the quickest way to check that your build
works before you add Pis.

## Repository layout

```text
acies-os/
├── docker/
│   ├── server/Dockerfile      # acies-server image (edge)
│   ├── pi/Dockerfile          # acies-pi image (arm64)
│   ├── edge.yml               # zenohd + gateway + ui (+ gps/tracker/mqtt profiles)
│   ├── edge-classifier.yml    # joint SPAR on edge
│   ├── replay.yml             # simulated nodes on edge (scenario 2)
│   ├── classifier.yml         # per-node classifiers on edge (scenario 2)
│   ├── pi.yml                 # live sensors on Pi (scenario 1)
│   ├── pi-replay.yml          # replay sensors on Pi (scenario 3)
│   ├── .env.edge.example
│   └── .env.pi.example
├── just/                      # recipes: docker.just, edge.just, pi.just, dev.just
├── packages/
│   ├── core/                  # acies.core: Zenoh transport, service runtime
│   ├── sensors/               # acies-geo, acies-mic, acies-replay, acies-gps
│   ├── classifiers/           # acies-vfm, acies-vfm-encoder, acies-vfm-head, acies-spar, …
│   ├── controllers/           # acies-gateway, acies-tracker, acies-mqtt; data/*.toml site configs
│   └── buffers/
├── scripts/update-pis.sh      # push weights/data/branch/.env to all Pis
└── justfile
```

## Hardware checklist

**Edge server**

* macOS on Apple Silicon (the reference setup, uses Colima) or Linux x86_64/arm64 with Docker.
* At least 16 GB RAM if you run scenario 2 with all eight simulated nodes and their classifiers.
* About 10 GB free disk for images, plus room for replay data (about 6 GB for the full set).

**Each sensor node**

* Raspberry Pi 4/5 running a **64-bit** OS (the Pi image is `linux/arm64`).
* Raspberry Shake board (geophone), on `/dev/serial0`.
* USB microphone array (the reference node uses a 6-channel ReSpeaker that shows up as
  `ArrayUAC10`).
* Wi-Fi antenna or Ethernet, and a microSD card of at least 32 GB.
* *Optional:* GPS receiver.
