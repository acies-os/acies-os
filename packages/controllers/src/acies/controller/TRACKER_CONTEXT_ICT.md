# ICT Tracker Context

This note is a handoff for continued work on the tracker logic in:

- `/home/kara4/demo/acies-os/packages/controllers/src/acies/controller/tracker.py`

The goal is to connect the current AciesOS tracker implementation to the raw ICT field dataset and to document what was already learned from the exploratory analysis work.

## Problem Summary

The current tracker in `tracker.py` is a 1-D road particle filter that:

- models vehicle state as road arc position `s` and speed `v`
- converts arc position to XY on a road polyline
- compares expected vs observed relative log-energy across sensors
- assumes closer sensors receive higher energy
- uses all-sensor raw energy as the measurement input

The open problem is to turn the raw ICT dataset into a practical measurement stream for this tracker and to decide how much of the measurement model should stay purely relative-energy-based versus use calibrated or thresholded logic.

## Relevant Tracker Assumptions in Current Code

The existing tracker logic in `tracker.py` currently assumes:

- a known road polyline with cumulative arc length
- known sensor GPS coordinates
- particle filter measurement update based on relative log-energy ratios
- sensor `0` is used as the measurement reference in `_observed_relative_log_energy`
- expected energy is based on a log-distance attenuation model
- measurement input is already reduced to per-node energy values

This means the missing work is not the particle filter itself. The missing work is the upstream transformation from raw `mic` and `geo` streams into a stable, physically meaningful `energy dict from sensor nodes`.

## ICT Dataset Used

Dataset root:

- `/home/tkimura4/data/2024-03-29-ICT`

Observed raw layout:

- runs: `0..7`
- sensor nodes present in files: `rs1`, `rs2`, `rs3`, `rs5`, `rs6`, `rs7`, `rs8`, `rs10`
- per-node modalities: `mic`, `geo`, `dis`
- per-run GPS: `run*_gps.parquet`
- metadata: `run_ids.parquet`, `sensor_location.parquet`

Important file meanings:

- `mic`: acoustic samples
- `geo`: seismic samples
- `dis`: derived distance reference table, not a sensing modality

## Raw Schema Findings

From direct inspection in the `foundationsense` environment:

### `mic` parquet

- columns: `timestamp`, `samples`, `sample_rate`
- `timestamp`: float seconds
- `samples`: scalar sample value, not a list/array per row
- `sample_rate`: present and observed as `16000`

Example:

- `timestamp` around `1.711738e+09`
- `samples` like `-2189`, `-2240`, `-2258`

### `geo` parquet

- columns: `timestamp`, `samples`, `channel`
- `timestamp`: float seconds
- `samples`: scalar sample value
- `channel`: observed values like `SH3`

### `dis` parquet

Not fully characterized in code here, but prior utility notebooks under `/home/tkimura4/data/utils` treat `_dis` as:

- a time-indexed per-sensor distance table
- derived from GPS traces and sensor locations
- containing one or more target-distance columns, sometimes with target lat/lon columns too

### `sensor_location.parquet`

Expected to contain:

- sensor ID
- latitude
- longitude

### `run_ids.parquet`

Expected to contain:

- `run_id`
- target label / vehicle label
- split info
- start and end times

## Utility Code and Prior Art

The most useful prior ICT preprocessing references were under:

- `/home/tkimura4/data/utils/2024-06-22-ict/parquet_to_segment.ipynb`
- `/home/tkimura4/data/utils/2025-09-21-ict/parquet_to_segment.ipynb`
- `/home/tkimura4/data/utils/postexp/nbs/2024-06-22-ICT.ipynb`

Key findings from those notebooks:

- raw sensor parquet was treated as normalized sample streams
- `dis` was generated from GPS plus sensor coordinates
- segmentation aligned `mic` and `geo` by overlapping timestamps
- ICT utility code used:
  - about `16000` samples for `mic` per 1-second segment
  - about `200` samples for `geo` per 1-second segment
- distance labels were intended to be computed per time segment, not just per whole run

This lines up well with the Acies tracker needs:

- we need windowed energy per sensor
- we need timestamp-aligned distance labels or validation references
- we likely want sensor-relative log-energy rather than absolute raw energy

## Analysis Work Completed So Far

Work was done in `FoundationSense` to prototype an exploratory energy-distance analysis notebook and later a script. The relevant files there are:

- `/home/kara4/FoundationSense/src/data_preprocess/visualization/ict_energy_distance_tracker_analysis.ipynb`
- `/home/kara4/FoundationSense/src/data_preprocess/visualization/ict_energy_distance_tracker_analysis.py`

What was established:

- `mic` and `geo` are the correct sensing modalities for tracker work
- `_dis` should be treated as a reference distance source, not as an input modality
- GPS plus `sensor_location` can be used to derive ground-truth distance to each node
- the raw dataset is large enough that naive first-pass feature extraction is expensive

## Performance / Data Processing Lessons

The raw dataset is heavy:

- `128` raw `mic` / `geo` files were identified for feature work
- combined size was about `7.2 GB`
- naive first-pass feature extraction across all files is expensive

What caused trouble:

- repeated full-file parquet scans
- expensive window generation on raw `mic` at `16 kHz`
- unnecessary feature computations beyond the core energy signal

What improved things:

- caching intermediate outputs
- reusing cached sampling summaries
- reducing feature extraction to only energy-related quantities
- downsampling `mic` before expensive processing
- using `sample_rate` directly instead of inferring everything from converted datetimes

## What Is Still Missing

The main unresolved engineering task is upstream measurement construction for `tracker.py`.

Concretely, we still need to decide and implement:

1. How to compute per-node energy from `mic` and `geo`

- RMS
- mean square energy
- log energy
- per-window aggregation

2. How to normalize energy across nodes and modalities

- raw per-node energy is unlikely to be directly comparable
- tracker code currently uses relative log-energy ratios
- this suggests producing stable per-node energies first, then forming relative log ratios in tracker space

3. Whether to use:

- `mic` only
- `geo` only
- fused `mic + geo`

4. How to map timestamps / windows into tracker updates

- 1-second non-overlapping or overlapping windows are a practical starting point
- tracker `dt` should match the effective measurement update cadence

5. How to order sensors for the relative-log-energy measurement vector

- current tracker uses sensor `0` as reference
- for ICT, node ordering needs to be explicit and stable
- probably use road-order or config-order rather than arbitrary dict order

## Recommended Next Steps in AciesOS

### 1. Keep tracker.py mostly intact

The particle filter structure is reasonable. The first work should be on measurement input quality, not on rewriting the filter.

### 2. Build a small offline adapter around the ICT dataset

Suggested scope:

- load selected runs and nodes
- compute per-window energy for `mic` and/or `geo`
- emit a per-timestep `energy dict` in the same shape the tracker expects
- compare tracker estimates against GPS-derived vehicle position

### 3. Start with a reduced experiment

Do not use the full raw dataset first. Start with:

- a few runs
- a few nodes
- one modality at a time
- 1-second windows

This should give fast iteration on the tracker measurement model.

### 4. Prefer relative-energy measurements over absolute calibration first

The current tracker already expects relative log-energy. That is a better fit for this dataset than trying to perfectly calibrate absolute signal strength across heterogeneous nodes.

### 5. Use `_dis` only as validation context

If `_dis` is easy to align for a given run, compare it against GPS-derived distance. But the tracker input path should not depend on `_dis`.

## Suggested Immediate Task Breakdown

If continuing development here, the next clean sequence would be:

1. Add a small offline ICT loader or test harness near the controller package.
2. Compute windowed `mic` energy only for a reduced subset.
3. Feed that into `RoadParticleFilter.update(...)` using a stable node order.
4. Compare estimated arc position against GPS-projected ground truth.
5. Only then add `geo` or fused measurements.

## Important Caveat

The prior exploratory analysis in `FoundationSense` did not complete a full end-to-end threshold study on the entire dataset within the available session time. The bottleneck was large-scale first-pass feature extraction over raw `mic`/`geo` parquet files.

So the correct takeaway is:

- the dataset structure is understood well enough to continue tracker work
- the tracker measurement direction is still valid
- but the final “best threshold” or “best modality fusion” result has not yet been established

## Most Relevant Paths

- Tracker implementation:
  - `/home/kara4/demo/acies-os/packages/controllers/src/acies/controller/tracker.py`

- This handoff note:
  - `/home/kara4/demo/acies-os/packages/controllers/src/acies/controller/TRACKER_CONTEXT_ICT.md`

- Raw dataset:
  - `/home/tkimura4/data/2024-03-29-ICT`

- Prior utility notebooks:
  - `/home/tkimura4/data/utils/2024-06-22-ict/parquet_to_segment.ipynb`
  - `/home/tkimura4/data/utils/2025-09-21-ict/parquet_to_segment.ipynb`
  - `/home/tkimura4/data/utils/postexp/nbs/2024-06-22-ICT.ipynb`

- Prior exploratory analysis code:
  - `/home/kara4/FoundationSense/src/data_preprocess/visualization/ict_energy_distance_tracker_analysis.ipynb`
  - `/home/kara4/FoundationSense/src/data_preprocess/visualization/ict_energy_distance_tracker_analysis.py`
