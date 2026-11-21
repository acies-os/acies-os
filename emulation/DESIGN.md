# Network emulation: design and internals

How the emulator in this directory is built, for people changing it. To run experiments, see
[README.md](README.md).

## Architecture

```text
host
┌──────────────────────────────────────────────────────────────────────────────┐
│ acies-mn container (privileged, Mininet)                                     │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ node namespaces (rs1 … rs10)       root namespace = "edge" 10.0.0.254    │ │
│ │ ┌───────────────────────┐          ┌───────────────────────────────────┐ │ │
│ │ │ rs3 10.0.0.3          │ TCLink   │ zenohd :7500 ◄── node routers     │ │ │
│ │ │  zenohd ──────────────┼── s1 ────┼─►      :7501 ◄── edge services    │ │ │
│ │ │  geo + mic replay     │ bridge   │        8 rx threads               │ │ │
│ │ │  [vfm encoder + head] │ tc/netem │          │ Zenoh                  │ │ │
│ │ └───────────────────────┘          │  ┌───────┼───────┬────────┐       │ │ │
│ │                                    │  ▼       ▼       ▼        ▼       │ │ │
│ │ topo.py (Mininet runner)           │ spar    gps   recorder gateway    │ │ │
│ │  builds topology                   │ SPAR   truth   JSONL    :8765 ◄───┼─┼─┼── browser
│ │  starts node services              │                  ▲                │ │ │
│ │  applies faults with tc            │                  │ emu/events     │ │ │
│ │  pub emu/events, emu/state ────────┼──────────────────┘ (and monitor)  │ │ │
│ │  sub emu/ctl ◄── fault.py          │ monitor :8780 ◄───────────────────┼─┼─┼── browser
│ │  sub **/ctl/notify/start_at        └───────────────────────────────────┘ │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│ ui: static Vue app, host network :5173 (browser loads it, then talks to      │
│     the gateway on 127.0.0.1:8765, published by acies-mn)                    │
│ analyze.py (offline) ◄── ~/.acies/emu/records/*.jsonl                        │
└──────────────────────────────────────────────────────────────────────────────┘
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

### The edge router freezes for 10 s when several nodes drop out

When three or more nodes black out together, data from the **healthy** nodes also stops
reaching the edge for about 10 s, starting 10 s after the blackout (when the dead nodes'
leases expire). Nothing is lost: it waits in TCP buffers and arrives in a burst. This is a
Zenoh bug that affects the real edge server too, not an emulation artefact
([eclipse-zenoh/zenoh#1101](https://github.com/eclipse-zenoh/zenoh/issues/1101)):

1. Zenoh's TCP links set `SO_LINGER` to 10 s (`TCP_LINGER_TIMEOUT`, fixed at compile time).
2. At lease expiry the router closes each dead node's socket synchronously, on one of its
   receive threads. If data for that node is still queued, `close()` blocks for 10 s.
3. `zenohd` has only 2 receive threads by default. Once both are stuck in `close()`, the
   router reads nothing from any link.

It needs data queued for the dead nodes when their leases expire, so the first blackout
right after a router restart is often clean. Measured on 2026-11-19 and 2026-11-20 with
`zenohd` 1.8.0 and 3-node blackouts:
* With the default 2 receive threads: a 9–10 s stall in 11 of 14 runs. The 3 clean runs were
  each the first blackout after a router restart.
* With 8 threads: no stall in 3 of 3 repeat runs, nor in a 5-node blackout.

The edge `zenohd` (`docker/edge.yml`, `just edge-zenohd`) therefore starts with 8 receive
threads by default. Keep the count above the number of nodes that can fail at once:

```bash
ZENOH_RUNTIME='(rx: (worker_threads: 16))' just emu-up-infra   # more nodes
ZENOH_RUNTIME='(rx: (worker_threads: 2))' just emu-up-infra    # zenohd's default, reproduces the bug
```

To see it yourself, start the edge with 2 receive threads as above, run two blackouts of three
nodes and check `just emu-analyze`; healthy nodes show a 10 s hole in delivery.
`cat /proc/1/task/*/wchan` in the edge `zenohd` container shows the stuck threads in
`wait_woken` during the freeze, and `cat /proc/1/task/*/comm` lists the `rx-*` threads.

## How each part is implemented

Everything that is specific to emulation lives in `emulation/`, `docker/emu-edge.yml` and
`just/emu.just`. The node and edge services are the normal acies-os ones, unmodified.

### Container and image (`emulation/Dockerfile`, `just emu-up-nodes`)

* `acies-mn` is the `acies-server` image plus Debian's `mininet`, `bridge-utils` and
  `net-tools`. Mininet's Python package is linked into `/opt/mininet-py` and put on
  `PYTHONPATH`, so `topo.py` can use Mininet and the acies venv's `zenoh` in one process.
* `just emu-up-nodes` runs it `--privileged` (Mininet creates namespaces, veth pairs, bridges and
  `tc` qdiscs), with `nofile` raised to 65536, `DATA_DIR`, `MODEL_DIR` and `~/.acies`
  mounted, and `emulation/` mounted read-only at `/emu`, so script edits need no rebuild.
  It publishes `127.0.0.1:8765` for the gateway that will later join its namespace.

### Topology (`topo.py`, `Emulator.build`)

* A star: one Linux bridge `s1` (`LinuxBridge`, no OpenFlow controller), one Mininet host
  per node (`10.0.0.<i>`, its own network namespace), and the `edge` host
  (`10.0.0.254`), created with `inNamespace=False` so it is the container's own namespace.
* Every link is a `TCLink`: a veth pair whose two ends are shaped with `tc`. Mininet puts an
  HTB rate limit (`bw`) and a `netem` qdisc (`delay`, `jitter`, `loss`, `max_queue_size`)
  on each end. Each end shapes the traffic it sends, so the node-side end shapes the uplink
  and the switch-side end the downlink: `delay` applies once per direction (RTT = 2 × delay).
* Link settings come from `[links.default]`, overridden per node by `[links.<node>]`. The
  `edge` link is unshaped unless `[links.edge]` is set; shaping it affects every node.

### Node services (`Emulator.start_nodes`)

* Each node's processes run inside its namespace (`host.popen`), from the image's venv,
  with `ACIES_NAMESPACE=<node>`, `ACIES_NAME=<service>` and `ACIES_CONNECT=tcp/127.0.0.1:7501`.
  To the rest of acies-os they look exactly like a Pi.
* `zenohd --listen tcp/127.0.0.1:7501 --connect tcp/10.0.0.254:7500` bridges the node to
  the edge router; this connection is the one crossing the shaped link.
* `acies-replay` for geo and mic with `--loop` and a far-future `--start-at` (2100-01-01), so
  they wait for a deploy. With `classifier = true`, `acies-vfm-encoder` and `acies-vfm-head`
  start `--deactivated`, as on a Pi, and a deploy activates them.
* Output goes to `~/.acies/emu/logs/<node>-<service>.log`. Processes that exit are logged
  every 5 s but not restarted.

### Edge stack (`docker/emu-edge.yml`)

* The normal `edge.yml` and `edge-classifier.yml` services, with
  `network_mode: container:acies-mn`, so they run in the edge namespace and listen on
  `10.0.0.254`. The emulated nodes reach them only through `s1` and their shaped links.
* `emu-edge.yml` also adds the recorder (from the `acies-mn` image) and binds the gateway on
  `0.0.0.0:8765` so the port published by `acies-mn` reaches it. The UI stays on the host
  network because it only serves static files.

### Deploys and timing

* A deploy (UI button or `deploy.py`) is a `ctl` frame on the gateway's WebSocket.
  `deploy.py` sends the same msgpack frame as the UI: `joint-nodes` mode with SPAR on every
  node, or `single-node` VibroFM with `--mode single`.
* The gateway configures each node and then sets `start_at` = now + 30 s on every replay.
  Each change is announced on `<namespace>/<service>/ctl/notify/start_at`.
* Replays share a data origin (`acies-replay --origin run`): the run's earliest timestamp
  plays at `start_at`, so all nodes play the same moment together.
* `topo.py` subscribes to `**/ctl/notify/start_at`. On a new value it arms the schedule:
  1. reset every link to the experiment file's settings;
  2. publish an `armed` event, carrying the event list, on `emu/events`;
  3. start a thread that sleeps until `start_at + at` for each event and applies it.

  A generation counter stops the previous deploy's thread, so redeploying restarts the schedule.

### Link changes (`Emulator.apply_event`)

`topo.py` stores each link's base settings, its current settings and a `down` flag, and
re-applies them with Mininet's `intf.config(...)` on both veth ends. That replaces the `tc`
qdiscs in place. A lock serializes all changes.

| Action | Implementation |
| --- | --- |
| `down` / `up` | Set or clear `down`; while it is set, `loss=100` overrides the current loss. The interface stays up, so TCP retransmits until the link returns or Zenoh's 10 s lease expires. |
| `set` | Merge the given values into the current settings and re-apply. |
| `reset` | Current settings = base settings, `down` cleared. |
| `unplug` / `plug` | `configLinkStatus` takes both veth ends down / up (`ifconfig`). `plug` re-applies shaping. |

### Live faults (`fault.py`, `just emu-fault`)

`fault.py` opens a Zenoh client session on the edge router and puts one msgpack message on
`emu/ctl`, for example `{"action": "set", "nodes": ["rs3"], "loss": 20}`. `topo.py`
subscribes to `emu/ctl` and applies it exactly like a scheduled event, tagged `source: ctl`.
Anything that can put on Zenoh, including a future gateway or UI, can send faults this way.

### Event log (`emu/events`)

Every applied change is published as msgpack on `emu/events`:

```text
source     "schedule" | "ctl" | "arm"
wall_ns    when it was applied
start_at   the armed replay start
at         seconds after start_at (scheduled events)
action     down | up | set | reset | unplug | plug | armed
nodes      links it applied to
params     the values given
effective  each link's resulting settings, e.g. {"rs3": {"bw": 10, "delay": "5ms", "loss": 100}}
events     (armed only) the experiment's event list
```

### Recorder (`recorder.py`)

An acies service (`AciesApp`) on the edge that appends one JSON line per message, stamped with
`wall_ns`, to `~/.acies/emu/records/<UTC time>.jsonl`. It subscribes to:

| Topic | Record | Content |
| --- | --- | --- |
| `**/spar` | `spar` | label, score, lat/lon, `window_start_s`, contributing `nodes`, `infer_ms` |
| `**/gps/truth` | `truth` | GPS truth per vehicle on the data clock (`data_ns`) |
| `**/energy` | `energy` | one per node, modality and 1-s window that reached the edge |
| `emu/events` | `event` | link changes and `armed` |

`energy` is used as the delivery signal: each replay publishes it with every 1-s window it
sends, over the same link as the raw data.

### Analysis (`analyze.py`, `just emu-analyze`)

1. Picks a deploy by its `armed` event and keeps the records from it until the next deploy.
2. Cuts the run into phases at every applied link change (`event` records from the schedule,
   the monitor page or `fault.py`; the deploy's own reset is not one). A change made before
   `start_at` counts as time 0, the starting state.
3. For SPAR, the time of a window is its data time minus the earliest data time seen (the
   shared origin), and its error is the distance to GPS truth at the window start (±1 s).
   Lag is output wall time minus the time the window's data finished playing.
4. Delivery is measured on the wall clock: the `energy` records that arrived during a phase,
   out of 2 per node per second.

### Link state (`emu/state`)

A thread in `topo.py` publishes every link's state once a second. For each namespace (the
container root for switch ports and the edge, `mnexec -a <pid>` for each node) it runs
`ip -s -j link` for byte counters and `tc -s -j qdisc` for the root qdisc's drops and backlog.
"Up" is what the node-side interface sends (for the edge link, the switch side), "down" the
other end. Rates are the change since the previous second.

```text
wall_ns, experiment, start_at, nodes, events (the schedule)
links.<name>: config {bw, delay, ...}, down, unplugged,
              up_kbps, down_kbps, up_drops_ps, down_drops_ps, up_backlog_b, down_backlog_b,
              up_bytes, down_bytes, up_drops, down_drops (counters)
```

### Monitor (`monitor.py`, `monitor.html`)

An edge service (`monitor` in `emu-edge.yml`) with a plain Zenoh client session. It subscribes
to `emu/state`, `emu/events`, `**/energy`, `**/spar` and `**/gps/truth`, and resets its run
state on each deploy (`armed`).

* **Replay time** of a SPAR window: its data time minus the run's origin, the data time that
  played at `start_at`. Every `energy` arrival has `data - wall = origin - start_at - delay`,
  so `origin = start_at + max(data - wall)` over the arrivals since the deploy. This matches
  `analyze.py` and still works when the monitor starts in the middle of a run (its
  history then starts at that point).
* **Error**: distance to the nearest vehicle's truth at the window start (±1 s). **Lag**:
  output wall time minus the time the window's data finished playing.
* **Delivery**: `energy` messages that arrived in the last 10 s, out of 2 per second.
* **Extra delay**: `max(data - wall)` minus a message's own `data - wall`, i.e. the delay on
  top of the fastest arrival; it needs no clock calibration. About 25 ms is normal (replays do
  not publish exactly on second boundaries).
* **Site**: like the gateway, it loads every `*.toml` in `packages/controllers/data` (mounted
  at `/config`) and shows the one the gateway is using. It reads that from the `map` field of
  the gateway's `health` messages (it keeps a WebSocket client open to the gateway), so a
  deploy from the main UI to another site switches the page too. Before the first deploy it
  shows `--map` (`ACIES_CONFIG`, default `ict.toml`).

* **Targets**: each site's `[routes]`; a target counts the site's nodes whose geo recording for
  that run is in `DATA_DIR` (mounted read-only at `/data`), and the page disables targets with
  none.
* **Truth on the map** ends `TRUTH_DELAY_S` = 2 s before the newest truth, as the main UI does
  (its `GT_CURRENT_DELAY_MS` is 2.5 s), so it lines up with SPAR's latest window.

It serves `monitor.html` (read on each request, so edits show on reload) and a WebSocket at
`/ws` with the `websockets` library. Every second each browser gets a JSON snapshot. The same
metrics, without the per-window history, go out as msgpack on `emu/metrics`, so the gateway
or another UI can use them. From the page it accepts `{"type": "fault", "action", "nodes",
bw/delay/jitter/loss}`, which it puts on `emu/ctl`; `{"type": "deploy", "target", "map", "plan"}`,
which first puts the plan on `emu/ctl` as `{"action": "plan", "events": [...]}` and then deploys
through `deploy.py`'s `deploy()` (joint SPAR on the emulated nodes); and `{"type": "plan"}`,
which replaces the running replay's schedule. `topo.py` validates a plan, replaces its schedule
with it, records a `plan` event (ignored by `analyze.py`), and, if a replay is armed, runs the
plan's remaining events against that replay's `start_at`.

The page builds the plan from rows (time, links, change, value, duration); a row with a duration
becomes the change plus its revert (`down`/`up`, `unplug`/`plug`, `set`/`reset`). The configuring
window opens on Deploy and counts down to the new `start_at` (the gateway's 30 s
`_REPLAY_AHEAD_TIME_S`, the time the testbed gets to configure and synchronize the nodes), for
deploys from the main UI as well.

## Status for UI work

Build, topology, scheduled and live faults, recording, offline analysis and the monitor
page all run. The existing UI works unchanged during emulation: map, SPAR track and
predictions. For a fuller network-emulation UI in the main `ui` app:

| Missing | Today | What it would take |
| --- | --- | --- |
| Live metrics in the main UI | on Zenoh (`emu/state`, `emu/metrics`, `emu/events`) and on the monitor page; the gateway does not forward them, and `ws://performance` still sends random numbers | gateway relays the `emu/*` topics as `ws://` topics; replace `ws://performance` with `emu/metrics`' SPAR numbers |
| Faults from the main UI | monitor page and `just emu-fault` | gateway forwards a WebSocket frame to `emu/ctl` (only when running under emulation) |
| SPAR late drops | logged by SPAR only | SPAR publishes its `late_drops` counter |
| Message spec | DESIGN.md and the code | a written, versioned schema per topic |
| Starting experiments and browsing past runs | command line, JSONL files | not designed yet |

