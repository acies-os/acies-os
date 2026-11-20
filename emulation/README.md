# Network emulation (Mininet)

Emulates the network between the sensor nodes and the edge server, so you can
inject transmission failures (blackouts, loss, delay, bandwidth limits) and
measure what they do to edge-side localization (SPAR) during a replay.

Everything runs on one machine in Docker. The host network is not modified.

## How it works

```text
 acies-mn container (privileged, Mininet)                          host
 ┌──────────────────────────────────────────────────────────────┐
 │ rs1 netns ─┐  each: zenohd (127.0.0.1:7501 -> 10.0.0.254:7500)│
 │ rs2 netns ─┤        geo + mic replay [+ vfm_encoder/vehicle]  │
 │   …        ├─ s1 ── edge 10.0.0.254 (container root namespace)│
 │ rs10 netns ┘  TCLink per node: bw/delay/jitter/loss via tc    │
 │                                                               │
 │ edge services (docker compose, network_mode: container:acies-mn)
 │   zenohd :7500/:7501, gateway :8765, gps, spar, recorder      │── 127.0.0.1:8765 ──► browser
 └──────────────────────────────────────────────────────────────┘    ui (host network) :5173
```

* **Nodes** are Mininet hosts (network namespaces) that run the same commands as a
  Pi in `docker/pi-replay.yml`, from the acies venv in the image. Each node has its
  own `zenohd` that bridges to the edge `zenohd`, exactly like the real deployment.
* **The edge** is the normal edge compose stack (`edge.yml` + `edge-classifier.yml`)
  with `docker/emu-edge.yml` on top, which puts it in the Mininet container's network
  namespace and adds the recorder.
* **Faults** come from an experiment file and are timed from the replay start: the
  emulator watches for the `start_at` that every deploy sends and re-arms the schedule.
  Live faults can be sent at any time with `just emu-fault`.
* **Replays idle** until a deploy (they start with a far-future `start_at`), play once,
  then wait for the next deploy (`--loop`).

## Quick start

Prerequisites: the normal setup (`.env`, data, models, `just build-server`, `just build-ui`)
and the SPAR weight in `MODEL_DIR`. Do not run the normal edge stack at the same time
(both use ports 8765 and 5173).

```bash
just emu-build                  # once, and after rebuilding acies-server
just emu-up outage              # emulator with emulation/experiments/outage.toml
just emu-edge-up                # edge stack (another terminal; Ctrl-C to stop)
just emu-deploy miata           # or Deploy in the UI: joint nodes, SPAR, target Miata
just emu-logs                   # watch the schedule fire
just emu-analyze                # after the replay (15 min for ICT run 2)
just emu-down                   # stop everything
```

`just emu-up smoke` is a 2.5-minute check with a single blackout.

## Experiment files

`emulation/experiments/*.toml`; see `outage.toml` for a commented example.

```toml
[nodes]
names = ["rs1", "rs2", "rs3", "rs5", "rs6", "rs7", "rs8", "rs10"]
classifier = true          # also run vfm_encoder + vfm_vehicle on each node

[links.default]            # every node <-> switch link
bw = 10                    # Mbit/s
delay = "5ms"              # one way, per interface: RTT is 2x
loss = 0                   # percent

[links.rs10]               # per-node override
delay = "40ms"

[links.edge]               # shared edge uplink (unshaped if omitted)
bw = 2

[[events]]
at = 120                   # seconds after replay start
nodes = ["rs3", "rs5"]     # name, list, "all", "edge", or "all+edge"
action = "down"
```

| Action | Effect |
| --- | --- |
| `down` / `up` | 100% loss in both directions; the interface stays up, like a radio blackout. Zenoh's TCP connection stalls and data published meanwhile is dropped. |
| `set` | Change `bw`, `delay`, `jitter`, `loss`, `max_queue_size` (merged into the current values). |
| `reset` | Back to the file's link config; clears `down`. Every deploy starts with a reset. |
| `unplug` / `plug` | Interface down / up. The node gets immediate "network unreachable" errors instead of silence. |

Changing the topology or link defaults needs `just emu-down` and `just emu-up` again.

## Live faults

```bash
just emu-fault rs3 down
just emu-fault rs3,rs5 set --loss 20 --delay 100ms
just emu-fault edge set --bw 1
just emu-fault all reset
```

Live changes are recorded like scheduled ones.

## What a failure does to SPAR

Raw geo (~0.8 KB/s) and mic (~32 KB/s) stream from every node to SPAR on the edge,
about 0.27 Mbit/s per node and 2.2 Mbit/s for eight. SPAR cuts the data clock into
2-second windows. A window is processed as soon as every active node stream has
delivered data up to the window's end, or at the latest `--max-wait` seconds (default 2,
wall clock) after the first data for the window's last second reached SPAR. A stream
that has been silent for 10 s (`STALE_STREAM_S`) is no longer waited for. So:

* A node whose data is **missing** is left out of the window after `max_wait` (its slot is
  zero with `valid=0`); the output's `nodes` list shows who contributed.
* A node that is **late** by more than `max_wait` is also left out: its data arrives after
  the window closed and is dropped. Over TCP, packet **loss** turns into retransmission
  **delay**, so moderate loss mostly shows up this way.
* A **bandwidth** cap below a node's rate makes its backlog grow until it is always late.
* A stream that delivered only one of the window's two seconds is left out too, rather
  than padded with silence.
* For the first 10 s that a node is missing, every window waits the full `max_wait`, so
  SPAR's output lag grows by that much but does not accumulate.

`max_wait` is the experiment knob for "how long does the edge wait for a slow node". Set it
with `acies-spar --max-wait`, or live with a kv set of `max_wait_s` on `joint/spar`.

### Things to know about the replay data

* Replays share a data origin: every node maps the earliest start among the run's files to
  `start_at` (`acies-replay --origin`, default `run`). Nodes whose files start late simply
  begin late, as in the field. In ICT run 2, **rs5 starts 113 s in**, and rs7's mic ends at
  629 s and its geo at 680 s. In run 3, rs7 starts 202 s in.
* The recordings have holes of their own, which show up as missing nodes even with perfect
  links. In ICT run 2: rs2's geo has about half its samples; rs3's and rs7's geo skip about
  60 seconds each; rs1, rs5, rs6 and rs7 each have a 200–290 s gap mid-run. Compare fault
  runs against `baseline`, not against "all 8 nodes".
* **Recovery after a blackout takes ~6 s** beyond the link coming back (Zenoh's 10 s lease
  expires during a longer outage, and the session has to reconnect). This is real behaviour
  worth measuring, not a bug.

## Recording and analysis

The recorder writes `~/.acies/emu/records/<UTC time>.jsonl` (owned by root; read it via
`just emu-shell`). It contains:

* `spar`: every localization (label, score, lat/lon, data window, contributing nodes)
* `truth`: GPS ground truth (same data clock as the sensors)
* `energy`: one per node, modality and second that reached the edge (delivery)
* `event`: every link change, and `armed` for each deploy

`just emu-analyze [file] [--deploy N]` prints, per phase between events: SPAR windows
produced out of those expected, mean contributing nodes, localization error (median,
mean, p90 metres), and the share of each node's data that reached the edge. It also
writes `<record>.windows.csv` with one row per SPAR output.

SPAR does not publish windows it classifies as background (no vehicle), so a phase
can produce fewer windows even with healthy links. Compare against `baseline`.

## Debugging

* `just emu-logs`: topology, deploys, applied faults.
* Node process output: `~/.acies/emu/logs/<node>-<service>.log`; acies logs as usual
  under `~/.acies/logs/<node>/`.
* `just emu-shell rs3` opens a shell in a node's namespace (`ping 10.0.0.254`,
  `tc -s qdisc`, `ss -tn`); `just emu-shell` opens the container.
* If `acies-mn` restarts, restart the edge stack too: it lives in that container's
  network namespace.

## Limits

* All nodes share one clock, so clock skew between nodes is not emulated.
* CPU is not emulated: nodes run on the server's cores, not at Pi speed.
* Links are wired (netem), not a Wi-Fi model. Mininet-WiFi would be needed for radio
  effects such as mobility or interference.
* The container is privileged. Membership in the `docker` group already amounts to
  root on the host, but check that this is acceptable on shared machines.
