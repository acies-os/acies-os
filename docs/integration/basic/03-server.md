# 3. Edge Server

The edge server builds all Docker images, hosts a small image registry for the devices, and runs
the edge services (zenohd, gps, gateway, ui).

## Install tools

* **Docker** with Buildx and Compose.
  * macOS (Apple Silicon): `just setup-docker-apple-silicon` installs Docker and Colima, then
    `just builder-setup` starts it. After a reboot, run `just docker-up`.
  * Linux: install Docker Engine (<https://docs.docker.com/engine/install/>), then run
    `docker buildx create --name acies-builder --use --driver-opt network=host`.
    On an **x86_64** machine, also run `docker run --privileged --rm tonistiigi/binfmt --install arm64`
    once so it can build the ARM image for the devices.
      - Need to install both `docker`, `docker-buildx`, and `docker-compose-v2`
* **just** (<https://just.systems>), **git**, **rsync**, **GNU parallel**.

## Clone

```bash
mkdir -p ~/acies && cd ~/acies
git clone git@github.com:acies-os/acies-os.git
git clone git@github.com:acies-os/ui.git
```

## Create `.env`

```bash
cd ~/acies/acies-os
cp docker/.env.edge.example .env
```

Change only these lines. Keep every other line as it is in the template.

```bash
PI_HOSTS="rs1 rs2 rs3"                     # SSH names of your devices

ZENOH_LISTEN=tcp/0.0.0.0:7500              # devices connect to this port
ACIES_LISTEN=ws://<EDGE_IP>:8765           # or ws://127.0.0.1:8765 if you only open the UI on this machine

SCENE=2024-03-29-ICT                       # recording to replay
RUN=2
GPS_LABEL=miata                            # vehicle in that run
DATA_DIR=/Users/<you>/acies/replay         # folder that contains 2024-03-29-ICT/

REGISTRY_EXTERNAL=<EDGE_IP>:5100           # where devices pull the image from
```

`<EDGE_IP>` is the edge server's LAN address. See [Network](05-network.md).

## Build the images

```bash
ssh-add -l                       # your GitHub key must be listed
just build-server                # -> acies-server:latest
(cd ../ui && just build-ui)      # -> acies-ui:latest
just registry-up                 # image registry on port 5100
just build-pi                    # -> builds acies-pi for ARM and pushes it to the registry
```

The first build takes 10–20 minutes.

## Check

```bash
docker images | grep acies       # acies-server, acies-ui
just registry-list               # acies-pi
```
