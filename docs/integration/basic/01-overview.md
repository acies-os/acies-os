# 1. Overview

## What runs where

```text
 Edge device (rs1)                                      Edge server
 ┌────────────────────────────────┐                    ┌──────────────────────────────────┐
 │ geo  (replay seismic)  ─┐      │                    │ zenohd   message router          │
 │ mic  (replay audio)    ─┤      │                    │ gps      replays the vehicle GPS │
 │ vfm_encoder            ─┼───►  | zenohd ── LAN ───► │ gateway  collects results        │
 │ vfm_vehicle (classifier)┘      │   (port 7500)      │ ui       web page (port 5173)    │
 └────────────────────────────────┘                    └──────────────────────────────────┘
 Edge device (rs2, rs3, ...) ── same ────────────────►        browser ──► http://<edge>:5173
```

**Edge server.** Runs from the `acies-server` and `acies-ui` Docker images.

| Service | What it does |
| --- | --- |
| `zenohd` | Message router. The edge devices connect to it. |
| `gps` | Replays the recorded GPS trace of the vehicle so the UI can show where it is. |
| `gateway` | Receives data and classification results from every device and serves them to the UI. |
| `ui` | Web page with the map, sensor readings, and classification results. |

**Each edge device.** Runs from the `acies-pi` Docker image.

| Service | What it does |
| --- | --- |
| `zenohd` | Local message router, connected to the edge server. |
| `geo`, `mic` | Replay the recorded seismic and audio data for this node. |
| `vfm_encoder` | Foundation-model encoder that turns geo + mic into an embedding. |
| `vfm_vehicle` | Classification head that predicts the vehicle type from the embedding. |

The classifiers start **switched off**. When you choose the **FM** pipeline in the UI and deploy
it, the gateway turns on `vfm_encoder` and `vfm_vehicle` on every device.

## Device names

Every edge device has a name such as `rs1`, `rs2`, … (set by `ACIES_NAMESPACE` in its `.env`).
The name decides:

* which recorded node's data the device replays, and
* where the device appears on the UI map. Node positions come from
  `packages/controllers/data/ict.toml`.

The recorded dataset has nodes `rs1`, `rs2`, `rs3`, `rs5`, `rs6`, `rs7`, `rs8`, `rs10`, so give
your devices those names. Each device needs a different name.

## Edge device requirements

The Raspberry Pi is only an example. Any device works if it runs:

* 64-bit ARM Linux (`aarch64`), because the device image is built for `linux/arm64`,
* Docker, and
* an SSH server.

The devices replay recorded data, so they need no sensors attached.
