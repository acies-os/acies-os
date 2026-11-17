# Edge Server Setup

This page prepares the edge server (laptop or workstation) to build all images and run the
edge services: `zenohd`, `acies-gateway`, the UI, and the optional tracker, GPS, and SPAR.

Finish [Repository access and SSH](02-access.md) first.

## 1. Install host tools

| Tool | Why | Install |
| --- | --- | --- |
| Docker Engine + Buildx + Compose v2 | Build and run all images | macOS: see below. Linux: <https://docs.docker.com/engine/install/> |
| `just` | Task runner for every recipe | `brew install just` / `apt install just` / <https://just.systems> |
| `git`, `rsync`, `curl`, `jq` | Clone, copy weights/data, query registry | package manager |
| GNU `parallel` | `just deploy-pi` runs SSH commands in parallel | `brew install parallel` / `apt install parallel` |
| `uv` *(optional)* | Run services outside Docker (`*-local` recipes) | <https://docs.astral.sh/uv/getting-started/installation/> |

### macOS (Apple Silicon)

The reference setup uses [Colima](https://github.com/abiosoft/colima) as the Docker daemon.
Two recipes do the one-time setup:

```bash
just setup-docker-apple-silicon   # brew install docker docker-buildx docker-compose colima + plugin links
just builder-setup                # colima start (4 GB) and create the 'acies-builder' buildx builder
```

After a reboot, start the daemon again with:

```bash
just docker-up                    # colima stop/start, prints status
```

:::{tip}
Running all eight simulated nodes with classifiers (scenario 2) needs more than 4 GB. Raise the
VM memory if containers get OOM-killed: `colima start --runtime docker -m 12 -c 6`.
:::

### Linux (x86_64 or arm64)

Install Docker Engine from the official repository, add yourself to the `docker` group, and
create the builder. Use `--driver-opt network=host` so the builder can push to the registry on
`localhost`:

```bash
sudo usermod -aG docker "$USER" && newgrp docker
docker buildx create --name acies-builder --use --bootstrap --driver-opt network=host
```

To cross-build the **arm64** Pi image on an **x86_64** host, also register QEMU emulation once:

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
```

(Apple Silicon and arm64 Linux hosts build the Pi image natively, with no QEMU.)

## 2. Clone the repositories

```bash
mkdir -p ~/acies && cd ~/acies
git clone git@github.com:acies-os/acies-os.git
git clone git@github.com:acies-os/ui.git

cd acies-os && git checkout mono-mobicom-model-sep   # or the branch your team uses
cd ../ui    && git checkout ui-integration           # UI branch that matches the gateway
```

## 3. Create the edge `.env`

Every recipe loads `.env` from the repository root (`set dotenv-load` in the `justfile`), and the
compose recipes pass it with `--env-file .env`.

```bash
cd ~/acies/acies-os
cp docker/.env.edge.example .env
```

Edit at least these values. Everything about IP addresses is explained in
[Network configuration](05-network.md).

```bash
# Pis you will deploy to (SSH aliases from ~/.ssh/config)
PI_HOSTS="rs1 rs2 rs3"

# Edge namespace (defaults to `hostname -s` when unset)
ACIES_NAMESPACE=my-laptop

# Zenoh: local loopback endpoint + LAN endpoint the Pis connect to
ZENOH_CONNECT=tcp/127.0.0.1:7501
ZENOH_LISTEN=tcp/0.0.0.0:7500

# Gateway: site config in packages/controllers/data/, and WebSocket address for the UI
ACIES_CONFIG=ict.toml              # ict.toml | wp.toml | gq.toml
ACIES_LISTEN=ws://127.0.0.1:8765   # use ws://<EDGE_LAN_IP>:8765 to open the UI from another machine

# Local paths for data and model weights (see "Data and model weights")
DATA_DIR=/Users/<you>/acies/replay
MODEL_DIR=/Users/<you>/acies/models

# Registry address the Pis pull from (edge LAN IP)
REGISTRY_EXTERNAL=<EDGE_LAN_IP>:5100
```

Leave the weight filenames and label lists as in the template unless you are using different
models. They must match files that exist in `MODEL_DIR`.

:::{important}
`just/docker.just` contains a hard-coded fallback for `REGISTRY_EXTERNAL` (a lab IP). If you
don't set `REGISTRY_EXTERNAL` in `.env`, the Pis will try to pull from that address and fail.
:::

## 4. Build the images

```bash
cd ~/acies/acies-os
ssh-add -l                    # your GitHub key must be listed (see Access page)

just build-server             # -> acies-server:latest, loaded into local Docker
cd ../ui && just build-ui     # -> acies-ui:latest
cd ../acies-os
docker images | grep acies    # acies-server and acies-ui should both appear
```

`build-server` builds for the host platform. The first build downloads PyTorch and the private
dependencies and takes 10–20 minutes. Later builds reuse the dependency layer unless `uv.lock`
changes.

The Pi image is built here too, but it's pushed to a registry instead of loaded locally. See
{ref}`Build and deploy <sec-build-pi-image>`.

## 5. Start a local image registry

The Pis pull `acies-pi` from a Docker registry on the edge server, on port **5100** (macOS
reserves 5000 for AirPlay).

```bash
just registry-up              # runs registry:2 as container 'acies-registry', restart unless-stopped
just check-port 5100          # should report "in use"
just registry-list            # lists images in the registry (empty until you push)
```

## 6. Smoke test (scenario 2, no Pis)

With data and weights in place ([Data and model weights](07-data-and-weights.md)), run the whole
system on the edge server alone:

```bash
# terminal 1: zenohd + gateway + UI + replayed GPS + tracker
just edge-up-infra --profile replay --profile tracker

# terminal 2: simulated nodes (replay) + per-node classifiers
just edge-up-nodes --profile rs1 --profile rs3
```

Open <http://localhost:5173>. Nodes `rs1` and `rs3` should appear on the map, and energy and
classification results should update. Stop everything with `just edge-down-all`.

## Running services without Docker (development)

For quick iteration you can run edge services directly with `uv`:

```bash
uv sync --all-packages              # or: just deps-dev
just edge-gateway-local --config packages/controllers/data/ict.toml
just edge-tracker-local
just edge-spar-local
```

A `zenohd` must still be running and listening on `ZENOH_CONNECT` (`just edge-zenohd` starts one
in Docker).

## Checklist

- [ ] `docker info` works and `docker buildx ls` shows `acies-builder`.
- [ ] `.env` exists in the repo root with `PI_HOSTS`, `DATA_DIR`, `MODEL_DIR`, and `REGISTRY_EXTERNAL` set.
- [ ] `docker images` lists `acies-server:latest` and `acies-ui:latest`.
- [ ] `just check-port 5100` reports that the registry is listening.
- [ ] Scenario 2 smoke test shows nodes in the UI.
