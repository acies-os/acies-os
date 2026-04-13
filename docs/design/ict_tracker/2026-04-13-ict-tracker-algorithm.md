# ICT Tracker Algorithm

Date: `2026-04-13`

## Purpose

This is the consolidated algorithm report for the ICT tracker work.

The goal is not precise continuous localization. The goal is:

- one active point at each time
- smooth loop-like motion
- no obvious jump-arounds or jump-backs
- simple enough to maintain and port

## Geometry Assumption

For ICT, the deployment is:

- `4` stations
- `2` sensors per station
- a loop corridor, not a straight open segment

The correct loop order is:

- `S1- -> S2- -> S3- -> S4- -> S4+ -> S3+ -> S2+ -> S1+`

This is why a simple 1-D station scalar alone is not enough.

## Final Architecture

The current best algorithm has three layers.

### 1. Energy Windowing

Input at each timestep is a cross-sensor energy vector from `mic` windows:

- one energy value per sensor
- currently `1 s` windows in the offline ICT evaluation

This gives a vector like:

- `[rs1_mic, rs2_mic, ..., rs10_mic]`

### 2. Discrete Observation Layer

The lower-layer observation source is still `hybrid_mic`.

That means:

1. compute pair-aware `mic` station estimates
2. if station confidence is low, fall back to `template_mic`
3. get coarse direction from `sensor_centroid_mic`

This lower layer is not the final displayed trajectory. It is only the observation prior for the continuity layer.

### 3. Point-Lattice Continuity Layer

Build a `40`-node loop lattice:

- `8` loop sensors
- `5` points per sensor

Each point is a discrete possible vehicle location on the loop.

For each point, learn a mean energy template from ICT:

- not raw audio waveform
- a mean cross-sensor energy signature for that point

At each timestep:

1. score every loop point by template similarity to the current energy vector
2. bias the score using the lower-layer `hybrid_mic` station/side estimate
3. run a continuity-constrained decoder on the loop graph

## Current Best Decoder

There are now two closely related best modes:

- `smoothed_hybrid_mic`: full offline smoother
- `fixedlag5_smoothed_hybrid_mic`: runtime-oriented bounded-lag smoother

It uses:

- the same loop-point emissions as the causal continuity tracker
- a full-path smoother over the loop lattice
- stronger penalties against large node jumps

Best shared parameters found so far:

- `max_step_nodes = 2`
- `hop_penalty = 0.7`
- `stay_penalty = 0.05`
- `reset_penalty = 4.0`
- `template_weight = 1.8`
- `anchor_weight = 1.0`
- `direction_bonus = 0.35`
- `direction_penalty = 0.25`

This is the current best continuity-first result.

## Why It Works Better

Raw `hybrid_mic` tends to:

- snap between coarse anchors
- switch sides incorrectly
- create geometric jump-backs

The smoothed loop decoder does better because it:

- reasons in point states, not just station states
- constrains movement to nearby loop points
- uses future evidence to resolve ambiguous windows
- suppresses isolated observation glitches

So the displayed path becomes a coherent sequence on the loop, not a sequence of disconnected anchor decisions.

## Current ICT Result

From [`continuity_summary.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13/continuity_layer/continuity_summary.csv):

| Mode | Joint Station+Side Acc | Mean XY Error (m) | Mean Step (m) | P95 Step (m) |
| --- | ---: | ---: | ---: | ---: |
| `hybrid_mic` | `0.325` | `64.96` | `33.29` | `159.83` |
| `causal_hybrid_mic` | `0.370` | `56.92` | `21.00` | `121.23` |
| `smoothed_hybrid_mic` | `0.418` | `48.02` | `7.66` | `27.04` |
| `fixedlag5_smoothed_hybrid_mic` | `0.419` | `47.93` | `10.68` | `29.70` |

For this project, the most important improvement is:

- the predicted point trace is far smoother
- large jumps were heavily reduced

## Delay Interpretation

There can be effective display delay because the smoother uses future evidence.

To make this visible without cluttering the XY figures, I added timing/progress plots in the compact report.

Those timing plots show:

- GT loop progress vs time
- predicted loop progress vs time
- a coarse estimated lag in seconds

This is the right place to inspect delay. The XY plots stay visually clean.

Important correction:

- the earlier delay estimate was too coarse
- it mixed true temporal lag with constant loop-position offset
- after correcting that, the representative-loop delay estimate for the selected report loops is effectively `0 s`

So the earlier apparent "mustang low delay, `gle350` high delay" conclusion was mostly an artifact of the old estimator.

## What Is Portable

The portable part of this tracker is the architecture:

- discrete road-point lattice
- energy-based point scoring
- continuity-constrained path decoding

The non-portable part is:

- the exact ICT lattice geometry
- the exact ICT point templates
- the exact loop transition topology

So for a new deployment later:

1. define the new geometry as discrete road points
2. collect calibration traces
3. learn point templates for that deployment
4. reuse the same continuity decoder structure

That applies both to:

- another vehicle on the same ICT loop
- a simpler single-side road deployment with fewer stations

For a simpler road-line deployment, the loop graph would just become a line graph or short chain graph.

## Current Limits

This algorithm still has limits:

- it is trained/evaluated on ICT-derived point templates
- some runs remain harder than others
- the current offline smoother uses the full sequence, not yet a bounded fixed-lag runtime approximation

Those are engineering follow-ups, not reasons to revert to the particle filter.

## Runtime Recommendation

If we were implementing the runtime design direction now, the correct abstraction is:

1. configurable discrete geometry
2. energy-to-point observation model
3. continuity decoder
4. optional bounded-lag smoothing for display

For an actual runtime-oriented ICT tracker today, the best practical choice is:

- `fixedlag5_smoothed_hybrid_mic`

Why:

- it approximates the offline smoother closely
- it respects the `5 s` to `6 s` delay budget
- it preserves the continuity gains

That is the architecture to carry forward from ICT.
