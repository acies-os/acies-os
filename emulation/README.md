# Network emulation (Mininet)

Replays a recorded run through an emulated network between the sensor nodes and the edge, so
you can break links (blackouts, loss, delay, bandwidth limits) while it plays and see what
that does to SPAR's localization. Everything runs on one server in Docker; the host network
is not touched. Internals: [DESIGN.md](DESIGN.md).

## Run an experiment

Prerequisites: the normal replay setup (`.env`, data, models, `just build-server`,
`just build-ui`). Stop the normal edge stack first; both use ports 8765 and 5173.

```bash
just emu-build      # once, and after each `just build-server`; also builds the monitor web app
just emu-up         # emulated nodes + edge stack; Ctrl-C stops the edge stack
                    # (again while running: reattaches to the logs, restarts nothing)
```

1. Open the network monitor: http://127.0.0.1:8780
2. In **Experiment**, pick the site and target.
3. Optionally plan network changes: each row is a time after the replay starts, the links, the
   change (blackout, unplug, loss, delay, jitter, bandwidth) and how long it lasts.
4. Click **Deploy SPAR**. A configuring window counts down while the nodes are set up and
   synchronized (about 30 s); then the replay plays and the plan fires on its own.
5. Change links live too, from the links table or by clicking a node on the map.
6. Afterwards, in another terminal:

```bash
just emu-analyze    # results per phase between your changes
just emu-down-all   # stop everything
```

From another computer, forward the ports first and open the same URL locally:

```bash
ssh -L 8780:127.0.0.1:8780 -L 8765:127.0.0.1:8765 -L 5173:127.0.0.1:5173 <server>
```

`just emu-up` runs these two, which you can also start separately (nodes first):

| Emulation | Normal replay | Starts |
| --- | --- | --- |
| `just emu-up-nodes` | `just edge-up-nodes` | the sensor nodes, inside the emulated network |
| `just emu-up-infra` | `just edge-up-infra --profile replay` | zenohd, gateway, GPS, SPAR, UI, plus the recorder and monitor |

## The monitor page

* **Experiment**: site, target, the plan, and **Deploy SPAR**. Targets are the main UI's (ICT:
  GLE350, the default, and CX30); ones without recordings on this server are greyed out. Presets
  fill in common plans. The plan is kept in your browser; **Apply to running replay** swaps it
  into a replay that is already playing. After Deploy, a configuring window counts down while
  the nodes are set up and synchronized.
* **Links**: each node's state, its link settings, and what is measured over the last 10 s:
  traffic, packets dropped by the emulator, share of its data that reached the edge, extra
  delay, time since its last data.
* **Fault buttons**: **Down** is a radio blackout (all packets lost, the link stays up);
  **Unplug** takes the interface down (the node gets errors at once). The form below the table
  sets bandwidth, delay, jitter or loss on one link, all nodes, or the edge uplink (which every
  node shares). **Reset** restores the starting settings.
* **Map**: nodes coloured by link state (click one to black it out or restore it), the vehicle's
  GPS truth and SPAR's last predictions. As in the main UI, the truth is drawn 2 s late so it
  lines up with SPAR, whose windows finish 2 s after they start.
* **SPAR performance**: localization error, nodes per window, windows produced and output
  lag, with charts over the whole replay and a dashed line at each link change.

SPAR predicts the vehicle's position at the start of each 2-s window, so its error is measured
against the GPS truth at that moment.

Only sites whose node names match the emulated nodes are offered (ICT and West Point, not GQ).
The page follows deploys from the main UI too. It needs internet for the fonts and map tiles.

## Results

Recording is automatic: `~/.acies/emu/records/<UTC start time>.jsonl` on the server, one file
each time the edge stack starts, holding every deploy since (owned by root; read it with `sudo`
or `just emu-shell`).

`just emu-analyze [file] [--deploy N]` (default: newest file, last deploy) splits the replay
into phases at every link change and prints, per phase:

* SPAR windows produced out of those expected, mean nodes per window, output lag
* localization error: median, mean and p90 metres
* share of each node's data that reached the edge

It also writes `<record>.windows.csv` next to the record, one row per SPAR output.

## Experiment files (optional)

The page's plan covers most experiments. An experiment file in `experiments/` is the same plan
written down, for runs from the command line; the page loads it as its starting plan.
`just emu-up` uses `baseline` (healthy links, no faults).

```bash
just emu-up smoke   # rs3, rs6, rs7 black out at 40-80 s
just emu-up outage  # several faults over the replay; commented example of every option
```

```toml
[links.default]            # every node's link
bw = 10                    # Mbit/s (raw geo + mic is ~0.27 Mbit/s per node)
delay = "5ms"              # each direction

[[events]]
at = 40                    # seconds after the replay starts
nodes = ["rs3", "rs5"]     # a node, a list, "all", "edge" or "all+edge"
action = "down"            # down/up, set (+ bw, delay, jitter, loss), reset, unplug/plug
```

Changing link defaults or nodes needs a restart (`just emu-down-all`, then `just emu-up`).

## Other sites

The monitor and gateway load every site in `packages/controllers/data/*.toml` and use the one
you deploy. The emulated nodes come from the experiment file: for a site other than ICT, copy
`baseline.toml`, set `[nodes] names` to that site's node names, and set `SCENE`/`RUN` in `.env`
to a run that has data for them.

## Command line

```bash
just emu-fault rs3 down                          # same as the page's buttons
just emu-fault rs3,rs5 set --loss 20 --delay 100ms
just emu-fault all reset
just emu-deploy                                  # deploy GLE350 without the page
just emu-logs                                    # emulator log: deploys and applied changes
just emu-shell rs3                               # shell in a node's network (ping, ss, tc)
```

Node process logs: `~/.acies/emu/logs/<node>-<service>.log`.

## Changing the web app

The monitor page is a Vue 3 + Vite app in `ui/` (same stack as the main UI); see
[ui/README.md](ui/README.md) for its structure. With Node 22 on the server:

```bash
just emu-ui-dev     # hot reload at http://127.0.0.1:5174, talking to the running monitor
just emu-ui-build   # build it; the monitor serves the new build on the next page load
```

Without Node, `just emu-build` builds it inside Docker (then restart the stack).

## Things to know

* **The recordings have gaps.** In GLE350 (ICT run 4), rs7's audio stops at 381 s and its
  seismic at 542 s, and rs2's seismic has about 40% of its samples. Other runs differ (in run 2,
  rs5 starts 113 s in). Gaps show up as missing nodes with healthy links, so compare fault runs
  with a run without faults, not with "all 8 nodes".
* **Loss becomes delay.** Data goes over TCP, so lost packets are resent and arrive late. SPAR
  waits up to 2 s for a late node, then leaves it out. Heavy loss (about 30%) can stall a node
  completely.
* **A blackout longer than 10 s drops the node's session.** It reconnects about 6 s after the
  link returns.
* If `acies-mn` restarts, restart the edge stack too. If the monitor restarts, its charts start
  over (the recording is unaffected).

## Limits

* All nodes share one clock and the server's CPU: no clock skew, no Pi-speed processing.
* Links are wired-style (`tc`/netem): loss is random per packet, not in bursts like radio fades,
  and both directions get the same settings.
* The `acies-mn` container is privileged; check that this is acceptable on shared machines.
