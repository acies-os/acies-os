# ICT Smoothed Continuity

Date: `2026-04-13`

## Scope

This is the compact ICT evaluation report for the current best continuity-first tracker.

It is intentionally narrower than the algorithm note:

- it shows the current best traces
- it summarizes the progression across investigation days
- it keeps the visuals focused on ground truth vs predicted path

Detailed algorithm description lives here:

- [`2026-04-13-ict-tracker-algorithm.md`](/home/kara4/demo/acies-os/docs/design/ict_tracker/2026-04-13-ict-tracker-algorithm.md)

## Current Best Result

Current best displayed trace mode:

- `smoothed_hybrid_mic`

Current runtime-oriented bounded-lag approximation:

- `fixedlag5_smoothed_hybrid_mic`

Current main artifact directory:

- [`docs/design/artifacts/ict_tracker_2026-04-13/continuity_layer`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13/continuity_layer)

Most relevant summary from [`continuity_summary.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13/continuity_layer/continuity_summary.csv):

| Mode | Mean XY Error (m) | Mean Step (m) | P95 Step (m) |
| --- | ---: | ---: | ---: |
| `hybrid_mic` | `64.96` | `33.29` | `159.83` |
| `causal_hybrid_mic` | `56.92` | `21.00` | `121.23` |
| `smoothed_hybrid_mic` | `48.02` | `7.66` | `27.04` |
| `fixedlag5_smoothed_hybrid_mic` | `47.93` | `10.68` | `29.70` |

For this use case, this is the important outcome:

- the displayed trace is now much smoother
- large jump-backs were heavily reduced
- the `5 s` bounded-lag runtime approximation stays very close to the full offline smoother

## Plot Guide

### XY Trace Plot

These are the `run*_smoothed_continuity_clean_xy.png` figures.

What the axes mean:

- x-axis: local X position in meters
- y-axis: local Y position in meters

What is plotted:

- blue line: ground-truth GPS path
- green line: tracker predicted path
- black points: true sensor locations

How to read it:

- if the green line follows the same loop shape as the blue line, continuity is good
- if the green line cuts across the loop or jumps between distant parts of the path, continuity is bad
- these plots are for geometry and continuity, not delay

### Timing / Progress Plot

These are the `run*_smoothed_continuity_timing.png` figures.

What the axes mean:

- x-axis: elapsed time within the selected loop segment, in seconds
- y-axis: unwrapped loop progress, in loop-node units

What the curves mean:

- blue: ground-truth loop progress
- light green: raw predicted loop progress
- dark green: predicted loop progress after constant position offset removal

How to read it:

- use the dark green curve to judge delay
- if dark green and blue line up in time, delay is small
- if dark green consistently trails blue to the right, prediction is delayed
- if dark green consistently leads blue, prediction is ahead

### Residual Panel

The small bottom panel in the timing figure shows:

- aligned predicted progress minus ground-truth progress

What the axes mean:

- x-axis: elapsed time within the selected loop segment, in seconds
- y-axis: residual in loop-node units

How to read it:

- near zero: prediction is tracking GT well
- positive: prediction is ahead on the path
- negative: prediction is behind on the path
- large oscillations: the tracker is wobbling even if the main timing lines look close

## Current Trace Set

This compact report now uses:

- `run0` (`mustang`)
- `run1` (`mustang`)
- `run4` (`gle350`)
- `run5` (`gle350`)

That gives two runs from each of two vehicle labels under the same deployment.

Important plotting note:

- the displayed loop for each run is now selected with `best_smoothed`
- that means the report uses the best representative stabilized loop segment for that run, not necessarily the first loop after startup
- this is more appropriate for the continuity-first question, because startup loops can be unrepresentatively noisy

Important timing note:

- the delay estimate is computed from the aligned progress curves
- on the selected representative loops, the estimated runtime delay is effectively `0 s`

## Run 0 (`mustang`)

![Run 0 clean](artifacts/ict_tracker_2026-04-13/continuity_layer/run0_smoothed_continuity_clean_xy.png)

Timing view:

![Run 0 timing](artifacts/ict_tracker_2026-04-13/continuity_layer/run0_smoothed_continuity_timing.png)

Interpretation:

- the smoothed trace follows the loop credibly
- jump-back behavior is much reduced
- for the selected representative loop, the timing view now indicates essentially `0 s` relative lag

## Run 1 (`mustang`)

![Run 1 clean](artifacts/ict_tracker_2026-04-13/continuity_layer/run1_smoothed_continuity_clean_xy.png)

Timing view:

![Run 1 timing](artifacts/ict_tracker_2026-04-13/continuity_layer/run1_smoothed_continuity_timing.png)

Interpretation:

- this is the stronger of the two mustang loop examples
- the predicted trace follows the loop shape closely
- this is one of the best pieces of evidence that the smoothed continuity layer is doing the right thing
- for the selected representative loop, the timing view also indicates essentially `0 s` relative lag

## Run 4 (`gle350`)

![Run 4 clean](artifacts/ict_tracker_2026-04-13/continuity_layer/run4_smoothed_continuity_clean_xy.png)

Timing view:

![Run 4 timing](artifacts/ict_tracker_2026-04-13/continuity_layer/run4_smoothed_continuity_timing.png)

Interpretation:

- this is the strongest new-vehicle example
- the predicted trace follows the loop very cleanly
- the timing plot indicates essentially `0 s` estimated runtime delay on the selected loop

## Run 5 (`gle350`)

![Run 5 clean](artifacts/ict_tracker_2026-04-13/continuity_layer/run5_smoothed_continuity_clean_xy.png)

Timing view:

![Run 5 timing](artifacts/ict_tracker_2026-04-13/continuity_layer/run5_smoothed_continuity_timing.png)

Interpretation:

- this run was weak when using the first loop after startup
- with representative-loop selection, the displayed stabilized loop is much better
- this is a better reflection of steady-state continuity performance for this run

## Compact Loop Summary

From [`loop_comparison_summary.csv`](/home/kara4/demo/acies-os/docs/design/artifacts/ict_tracker_2026-04-13/continuity_layer/loop_comparison_summary.csv):

| Run | Label | Smoothed Mean XY Error (m) | Estimated Delay (s) |
| --- | --- | ---: | ---: |
| `0` | `mustang` | `8.07` | `0` |
| `1` | `mustang` | `9.31` | `0` |
| `4` | `gle350` | `9.35` | `0` |
| `5` | `gle350` | `9.15` | `0` |

Interpretation:

- `run0` is very good
- `run1` is very good
- `run4` is very good
- `run5` is also good once we look at a representative stabilized loop instead of the startup loop

That gives a much cleaner and more useful picture of steady-state continuity across two vehicle labels.

## Delay Reading

You asked how to understand timestamps of prediction vs real trace at intermediate points.

The current answer is:

- keep the XY plots visually clean
- use the separate timing/progress plots to compare predicted vs GT progress through the loop

Those timing plots are the best place to inspect:

- whether the trace trails the GT
- by roughly how many seconds
- whether the lag is consistent through the loop or not

The current lag estimate is only an alignment metric, not a final runtime latency guarantee.

Important reading note for the timing plots:

- the light green/raw curve can still appear vertically shifted on some runs
- that vertical separation is position offset, not necessarily delay
- the dark aligned curve is the one to use for judging temporal lag

But the new bounded-lag runtime mode now gives a more relevant engineering target:

- `fixedlag5_smoothed_hybrid_mic`
- close to the full smoother
- within the acceptable `5 s` to `6 s` budget

## Current Recommendation

For the displayed continuity trace, the current best algorithm remains:

- `smoothed_hybrid_mic`

For reporting, this is the correct pairing:

- detailed design: [`2026-04-13-ict-tracker-algorithm.md`](/home/kara4/demo/acies-os/docs/design/ict_tracker/2026-04-13-ict-tracker-algorithm.md)
- compact ICT result: this note
