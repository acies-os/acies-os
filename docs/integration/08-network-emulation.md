# 8. Network Emulation

The network emulator runs all eight edge devices **on the edge server**, connected to it
through an emulated network that you can break while a replay plays: black out a device, add
loss or delay, or cap the bandwidth. A web page shows the network conditions next to SPAR's
localization error, so you can see how failures affect the results.

Use it to test the system without any devices, or to run network-failure experiments. It only
needs the edge server; pages 4–6 are not required.

## What runs where

```text
 Edge server
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ acies-mn container (Mininet)                                             │
 │ ┌───────────────────────┐   emulated links   ┌────────────────────────┐  │
 │ │ rs1 … rs10            │   (loss, delay,    │ zenohd  gateway  gps   │  │
 │ │  zenohd, geo, mic     ├───── bandwidth, ──►│ spar    recorder       │  │
 │ │  (one per node)       │    blackouts)      │ monitor (port 8780)    │  │
 │ └───────────────────────┘                    └────────────────────────┘  │
 └──────────────────────────────────────────────────────────────────────────┘
        browser ──► http://<edge>:8780  (network monitor)
        browser ──► http://<edge>:5173  (main UI, unchanged)
```

Each emulated device runs the same services as a real one and connects to the edge `zenohd`
in the same way. The edge server runs its usual services, plus SPAR, which localizes the
vehicle from all eight devices' raw data. The commands mirror the ones on the earlier pages:

| Emulation | Without emulation | Starts |
| --- | --- | --- |
| `just emu-up-nodes` | `just edge-up-nodes` (or `just pi-up-replay` on each device) | the eight devices, inside the emulated network |
| `just emu-up-infra` | `just edge-up-infra --profile replay` | the edge services, plus SPAR, the recorder and the monitor |
| `just emu-up` | – | both, in the right order |

## 1. Prepare the edge server

Complete [3. Edge server](03-server.md) (tools, `.env`, `just build-server`, UI build). Then:

* **Data**: the geo and mic files of **all eight nodes** for your run, plus the GPS file, in
  `<DATA_DIR>/2024-03-29-ICT/` (see [7. Data and weights](07-data-and-weights.md)).
* **SPAR weight**: the file named by `SPAR_WEIGHT` in `.env` (`spar_2024-03-29-ICT.pt` in the
  template) in `MODEL_DIR`, from the model weights download.

Build the emulator image once, and again after each `just build-server`:

```bash
just emu-build
```

Stop the normal edge services first (`just edge-down-all`): both use ports 8765 and 5173.

## 2. Start

```bash
just emu-up
```

This starts the emulated devices with healthy links, then the edge services in the
foreground. Leave it running; `Ctrl-C` stops the edge services.

## 3. Run an experiment (browser)

1. Open `http://<EDGE_IP>:8780` (or `http://localhost:8780` on the edge server). From another
   computer, forward the ports first:

   ```bash
   ssh -L 8780:127.0.0.1:8780 -L 8765:127.0.0.1:8765 -L 5173:127.0.0.1:5173 <edge server>
   ```

2. In **Experiment**, pick the site and target (default `ICT`, `GLE350`, which is run 4). Targets
   without recordings on the edge server are greyed out.
3. Optionally plan network changes: each row says when (seconds after the replay starts), which
   devices, what change (blackout, unplug, loss, delay, jitter, bandwidth) and for how long. The
   presets fill in common plans.
4. Click **Deploy SPAR**. As in the main UI, a configuring window counts down about 30 s while
   the devices are configured and their start is synchronized. Then the replay plays and the
   planned changes happen on their own.
5. You can also change the network live: the **Down** and **Unplug** buttons in the table (or a
   click on a device on the map), and the form under the table for bandwidth, delay, jitter or
   loss. **Reset** restores healthy links.
6. Watch the effect: each device's delivered data and delay in the table, and SPAR's error,
   nodes per window and output lag below the map.

SPAR predicts the vehicle's position at the start of each 2-second window, so its error is the
distance from the GPS truth at that moment.

## 4. Results

Every run is recorded automatically on the edge server, in
`~/.acies/emu/records/<UTC time>.jsonl`. After a run, summarize it:

```bash
just emu-analyze
```

It splits the replay at each network change you made and prints, per phase: SPAR windows
produced, nodes per window, output lag, localization error (median, mean, p90 metres), and the
share of each device's data that reached the edge. It also writes a CSV with one row per SPAR
output next to the record.

## Stop

```bash
just emu-down-all
```

## Experiment files

The plan on the page can also be written in a file, for runs from the command line; the page
loads it as its starting plan:

```bash
just emu-up smoke      # rs3, rs6 and rs7 black out from 40 s to 80 s
just emu-up outage     # several failures; a commented example of every option
```

The files are in `emulation/experiments/`. Copy one to make your own.

## When something doesn't work

| Symptom | Check |
| --- | --- |
| Page says **Disconnected** | The edge services are running: `docker ps` lists `edge-emu-monitor-1`. |
| Map stays grey | The browser needs internet for the map tiles, as in the main UI. |
| A device shows 0% delivered with a healthy link | Its recording has no data at that time. In GLE350 (run 4), `rs7` stops early (audio at 381 s, seismic at 542 s), so compare failure runs with a run without failures. |
| Deploy does nothing | `just emu-logs` shows the emulator's log; device logs are in `~/.acies/emu/logs/`. |

More detail, including how each part works, is in `emulation/README.md` and
`emulation/DESIGN.md` in the repository.
