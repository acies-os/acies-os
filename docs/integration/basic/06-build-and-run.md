# 6. Build and Run

Before starting, complete pages 3–5 and put the data and weights in place
([page 7](07-data-and-weights.md)).

## 1. Build and publish the images (edge server)

Skip this if nothing changed since your last build.

```bash
cd ~/acies/acies-os
ssh-add -l
just build-server
(cd ../ui && just build-ui)
just registry-up
just build-pi
```

## 2. Send the image to the devices (edge server)

```bash
just deploy-pi                   # every device in PI_HOSTS pulls acies-pi:latest
```

## 3. Start the edge server

```bash
just edge-up-infra --profile replay
```

This starts `zenohd`, `gateway`, `ui`, and `gps` (the `replay` profile). Leave it running.

## 4. Start each device

```bash
ssh rs1
cd /ws/acies/acies-os
just pi-up-replay
```

This starts the five services needed for vehicle classification (`zenohd`, `geo`, `mic`,
`vfm_encoder`, `vfm_vehicle`). Leave it running.

## 5. Turn on classification (browser)

1. Open `http://<EDGE_IP>:5173` (or `http://localhost:5173` on the edge server).
2. Each device should appear on the map with live sensor readings.
3. In the configuration panel, choose the **FM** pipeline and deploy. The gateway turns on
   `vfm_encoder` and `vfm_vehicle` on every device, and vehicle predictions start appearing.

## Stop

```bash
just edge-down-all                                              # edge server
ssh rs1 'cd /ws/acies/acies-os && just pi-down'                # each device
```

## When something doesn't work

```bash
docker compose -p edge-replay logs -f gateway                   # edge server
ssh rs1 'cd /ws/acies/acies-os && docker compose -p pi logs -f' # device
```

Logs are also written to `~/.acies/logs/` on every machine. For connection problems, see
[Network](05-network.md).
