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
DATA_DIR=/home/<you>/acies/replay          # folder that contains 2024-03-29-ICT/ (absolute path, no ~)
MODEL_DIR=/home/<you>/acies/models         # folder that contains the .pt weight files (absolute path, no ~)

REGISTRY_EXTERNAL=<EDGE_IP>:5100           # where devices pull the image from
```

`<EDGE_IP>` is the edge server's LAN address. See [Network](05-network.md).

> [!IMPORTANT]
> `DATA_DIR` and `MODEL_DIR` are folders **on the edge server**. Inside the containers they always
> appear as `/data` and `/model`, so log messages show paths like `/data/2024-03-29-ICT/...` or
> `/model/Parkland_ict2024_encoder.pt` even when your files are somewhere else. If a log says such a
> file does not exist, the file is missing from your `DATA_DIR` / `MODEL_DIR`.

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

## Test the server on its own

Before setting up any edge device, you can run the whole system on the edge server alone. The
server simulates one edge device (`rs1`) by replaying its data and running its classifier locally.

**1. Put the files in place** (download links in [Data and model weights](07-data-and-weights.md)). An example file structure looks like this. 

```text
acies
├── acies-os
├── models
│   ├── Parkland_ict2024_class_layer.pt
│   ├── Parkland_ict2024_encoder.pt
│   └── Parkland_wp2026_class_layer.pt
├── replay
│   └── 2024-03-29-ICT
│       ├── runX_Y_gps.parquet
│       ├── ......
│       └── runX_rsZ_MOD.parquet
└── ui
```

All five files must be listed with no "No such file" errors.

**2. Start the edge services** (terminal 1):

```bash
just edge-up-infra --profile replay      # zenohd, gateway, ui, gps
```

**3. Start the simulated node** (terminal 2):

```bash
just edge-up-nodes --profile rs1         # rs1 geo/mic replay + FM vehicle classifier
```

**4. Check in the browser.** Open `http://localhost:5173` (or `http://<EDGE_IP>:5173`). Node
`rs1` should appear on the map with live readings and the vehicle's GPS position. Choose the
**FM** pipeline in the configuration panel and deploy; vehicle predictions for `rs1` should
appear.

- For ssh user, simply `ssh -L 5173:localhost:5173 -L 8765:localhost:8765 SERVER` for port forwarding. 

**5. Stop everything** before moving on to real devices. A real device named `rs1` would clash
with the simulated one.

```bash
just edge-down-all
```

