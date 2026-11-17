# Build, Deploy, and Run

This page puts the pieces together: build every image, ship the Pi image to the nodes, and start
each scenario. Run all commands from the `acies-os` repo root unless noted.

## Build order at a glance

```text
edge server                                        each Pi
───────────                                        ───────
1. ssh-add (GitHub key in agent)
2. just build-server     → acies-server:latest
3. (ui repo) just build-ui → acies-ui:latest
4. just registry-up      → registry on :5100
5. just build-pi         → push acies-pi to :5100
6. just registry-trust <pi>   ───────────────────► /etc/docker/daemon.json
7. just deploy-pi             ───────────────────► docker pull + tag acies-pi:latest
8. scripts/update-pis.sh      ───────────────────► /model, /data/audrok, git pull, .env keys
9. just edge-up-infra ...                          10. just pi-up ... / just pi-up-replay
```

## Build the edge images

```bash
ssh-add -l                       # GitHub key loaded (see Access page)
just build-server                # acies-server:latest (host platform)
(cd ../ui && just build-ui)      # acies-ui:latest
```

(sec-build-pi-image)=

## Build and publish the Pi image

```bash
just registry-up                 # once; keeps running across reboots
just build-pi                    # buildx --platform linux/arm64 --push -> localhost:5100/acies-pi:latest
just registry-list               # should list acies-pi
```

`build-pi` pushes to `REGISTRY` (default `localhost:5100`), and the Pis pull from
`REGISTRY_EXTERNAL` (`<EDGE_LAN_IP>:5100`). These are the same registry under two names.

To build a tagged version instead of `latest`, run `TAG=v0.3.0 just build-pi` and
`TAG=v0.3.0 just deploy-pi`.

## Deploy to the Pis

```bash
just registry-trust rs1          # once per Pi (repeat for each host)
just deploy-pi                   # pull + tag on every host in PI_HOSTS, in parallel
scripts/update-pis.sh            # weights -> /model, drone data -> /data/audrok,
                                 # git checkout BRANCH on Pi, add missing .env keys
```

`scripts/update-pis.sh` options:

| Setting | Default | Meaning |
| --- | --- | --- |
| positional args | `PI_HOSTS` | Only update the listed Pis, e.g. `scripts/update-pis.sh rs2 rs3` |
| `--dry-run` | off | Print every command without changing anything |
| `MODEL_SRC` | `~/acies/models` | Local folder with weight files |
| `BRANCH` | `mono-mobicom-model-sep` | Branch checked out on each Pi |
| `PI_APP_DIR` | `/ws/acies/acies-os` | Repo path on the Pis |

:::{warning}
`update-pis.sh` runs `git checkout .` on each Pi, which **discards local edits** in the Pi's
checkout (but not its untracked `.env`). Commit or copy anything you changed on a Pi first.
:::

## Run: Scenario 1, field (live sensors)

**Edge server:**

```bash
just edge-up-infra --profile live --profile tracker
#   live    -> MQTT GPS bridge (needs MQTT_BROKER)
#   tracker -> vehicle tracker
#   add --profile spar for the joint SPAR classifier
```

**Each Pi** (`ssh rs1`, then `cd /ws/acies/acies-os`):

```bash
just pi-up --profile multimodal      # zenohd + geo + mic + e2e_vehicle
```

`docker/pi.yml` profiles: `geo`, `mic`, `geo-classifier`, `mic-classifier`, `multimodal`. You
must pass at least one; with no profile, nothing starts.

## Run: Scenario 2, lab (everything on the edge)

No Pis are needed. Replay data must be under `DATA_DIR` on the edge.

```bash
# terminal 1
just edge-up-infra --profile replay --profile tracker

# terminal 2: pick nodes with --profile rsN, or --profile all
just edge-up-nodes --profile rs1 --profile rs2 --profile rs3
```

`SCENE`, `RUN`, and `GPS_LABEL` in `.env` select the recording (for example
`2024-03-29-ICT`, run `2`, `miata`). To replay a dataset whose node names differ from `rsN`
(for example the GQ set), map them with `NODE_RS1=gq-1` and so on.

## Run: Scenario 3, tabletop demo (Pis replay recorded data)

Each Pi needs the scene under `/data` (for example `/data/2024-03-29-ICT/run2_rs1_*.parquet`) and
`/data/audrok/`. Copy them with `rsync` or `scripts/update-pis.sh`.

**Edge server:**

```bash
just edge-up-infra --profile replay --profile tracker
```

**Each Pi:**

```bash
just pi-up-replay                    # zenohd + geo/mic/drone_mic replay + all classifier variants (deactivated)
```

Open the UI, choose the pipeline ("fm" or "e2e", with or without drone) in the config panel, and
deploy. The gateway activates the matching models on every node.

## Verify a running system

```bash
# containers
docker compose -p edge-replay ps                 # edge
ssh rs1 'docker compose -p pi ps'                # Pi

# logs (any machine)
ls ~/.acies/logs/
tail -F ~/.acies/logs/<namespace>/<service>.log

# live container output
docker compose -p edge-replay logs -f gateway
ssh rs1 'docker compose -p pi logs -f zenohd'
```

Enable more logging by setting `ACIES_LOG=acies=DEBUG` (or a single module, for example
`acies.core=DEBUG`) in `.env`, then restart.

## Stop, update, rebuild

| Task | Command |
| --- | --- |
| Stop edge stack | `just edge-down-all` |
| Stop edge + prune dangling images | `just edge-clear` |
| Pull latest code + rebuild edge image | `just edge-rebuild` |
| Stop Pi stack | `ssh rs1 'cd /ws/acies/acies-os && just pi-down'` |
| Rebuild and roll out Pi image | `just build-pi && just deploy-pi`, then restart the Pi stacks |
| Shell inside the Pi image | `just pi-shell` (on the Pi; `just pi-shell true` mounts the repo for dev) |
| Run one Pi service by hand | `just pi-zenohd`, `just pi-geo`, `just pi-mic`, `just pi-vfm` |

## Checklist

- [ ] `just registry-list` shows `acies-pi`.
- [ ] `just deploy-pi` finishes without errors for every host.
- [ ] Scenario 2 works on the edge alone.
- [ ] In scenario 1 or 3, every Pi appears in the UI with live energy readings.
- [ ] Selecting a pipeline in the UI produces classification results.
