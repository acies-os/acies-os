# Integration Guide

This guide walks you through standing up a complete Acies-OS deployment on **your own hardware**:
one edge server (a laptop or workstation) plus one or more Raspberry Pi sensor nodes.

Read the pages in order the first time. Each page ends with a short checklist so you can confirm
that step worked before moving on.

```{toctree}
:maxdepth: 1
:numbered:

01-overview
02-access
03-server
04-pi
05-network
06-build-and-run
07-data-and-weights
```

:::{note}
**Which branch?** These instructions describe the containerised "mono-repo" layout of
`acies-os` (directories `docker/`, `just/`, `packages/`). At the time of writing, that layout
lives on the `mono-mobicom-model-sep` branch. Check out that branch (or whichever branch your
team tells you is current) on **both** the edge server and every Pi.
:::

## Quick reference

| What | Where |
| --- | --- |
| Edge-server image | `acies-server:latest` (built from `docker/server/Dockerfile`) |
| Pi image | `acies-pi:latest` (built from `docker/pi/Dockerfile`, `linux/arm64`) |
| UI image | `acies-ui:latest` (built from the separate `acies-os/ui` repository) |
| Env template, edge | `docker/.env.edge.example` → copy to `.env` in the repo root |
| Env template, Pi | `docker/.env.pi.example` → copy to `.env` in the repo root on each Pi |
| Repo path on Pi | `/ws/acies/acies-os` (override with `PI_APP_DIR`) |
| Model weights on Pi | `/model/` |
| Sensor/replay data on Pi | `/data/` |
| Logs (all machines) | `~/.acies/logs/<namespace>/<name>.log` |
