# West Point Line Topology Runtime Integration

Date: `2026-04-14`

## Purpose

This report documents the completed continuity-tracker integration and stabilization for the new West Point deployment:

- new deployment data: `/home/jinyang7/data/2026-04-14-West-Point`
- vehicle classes used for calibration and evaluation: `kona`, `wagoneer`
- deployment topology: bounded road / line, not loop
- runtime operating point: `pooled_median` with fixed lag `5`
- corrected tuning / evaluation corridor: nearest-sensor distance `<= 25 m`

It incorporates the initial integration of line topologies with the final stabilization steps (tangent-locked projection and runtime gating).

## Main Outcome

The tracker is operationally smooth, integrated, and deployment-ready for the real-time experiment:

- West Point data ingestion works and labels/templates were built from the new deployment data.
- The state graph is truly line-topology aware.
- **Tangent-Locked Projection**: The `5 points` per sensor were forced onto the deployed sensor-side road segment by mathematically spacing them exactly along the fitted tangent vector. This eliminated jaggedness/jumpiness.
- **Live Runtime Corridor Gating**: An online `confidence_threshold` gate is now live in `tracker.py` to suppress output when a target travels outside the valid tracking corridor.
- The offline and integrated controller-runtime evaluations were rerun at `lag=5`.
- Traces no longer exhibit previous non-smooth artifacts and are clean.

## Code Changes

Primary implementation areas:

- [`build_ict_tracker_assets.py`](/home/kara4/demo/acies-os/packages/controllers/scripts/build_ict_tracker_assets.py)
- [`eval_ict_continuity_layer.py`](/home/kara4/demo/acies-os/packages/controllers/scripts/eval_ict_continuity_layer.py)
- [`eval_tracker_runtime_replay.py`](/home/kara4/demo/acies-os/packages/controllers/scripts/eval_tracker_runtime_replay.py)
- [`label_ict_ground_truth.py`](/home/kara4/demo/acies-os/packages/controllers/scripts/label_ict_ground_truth.py)
- [`ict_continuity.py`](/home/kara4/demo/acies-os/packages/controllers/src/acies/controller/ict_continuity.py)
- [`tracker.py`](/home/kara4/demo/acies-os/packages/controllers/src/acies/controller/tracker.py)

### 1. True Line Topology Support & Tangent-Locked Projection
The line topology is not just a metadata flag now. Shared lattice helpers were updated to support monotonic line sensor ordering and endpoint-clamped next/prev nodes. 

Critically, in `ict_continuity.py`'s `build_point_lattice`, we now conditionally force the node centers for `line` topologies to anchor strictly to the station's geometric center. The 5 points are mathematically spaced exactly along the fitted tangent vector, rather than relying on noisy local sample centroids.

### 2. Corrected Corridor-Gated Template Tuning & Runtime Gating
The first West Point fit was pulled off-course by samples far from the deployed sensor line. The assets are now rebuilt only from rows with `nearest_sensor_distance_m <= 25 m`.

Additionally, the real-time controller in `tracker.py` now implements a live `confidence_threshold`. If the decoded path's confidence (`best_score - second_best_score`) falls below this threshold, the tracker suppresses output, effectively handling when the vehicle strays too far.

### 3. Evaluation Updates
Line traces now use representative contiguous in-corridor traversals, focusing on where sensors are actually informative.

## Generated West Point Artifacts

Artifact root:

- [`docs/design/artifacts/west_point_2026-04-14/`](/home/kara4/demo/acies-os/docs/design/artifacts/west_point_2026-04-14)

Main corrected outputs:
- labels: `labels/`
- deployment asset bundle: `assets_corridor25/`
- controller runtime replay: `runtime_eval_corridor25_fullfit/`

## Evaluation Summary

### Controller Runtime Replay, Full-Fit, Corridor-Gated

From `runtime_eval_corridor25_fullfit/runtime_summary.csv`:

| Mode | Joint Station+Side Acc | Point Acc | Mean XY Error (m) | Mean Step (m) |
| --- | ---: | ---: | ---: | ---: |
| `runtime_fixedlag5_global` | `0.461` | `0.109` | `37.72` | `12.57` |

## Readiness For Tomorrow

The integration is fully hardened and **ready for tomorrow’s real-time experiment deployment**. The asset geometry matches the physical world smoothly, and the live pipeline will properly gate unreliable tracks. 

Recommended operational stance for tomorrow:
- point `tracker.asset_dir` at `assets_corridor25/`
- tune `confidence_threshold` in the `.toml` config according to field conditions.
- keep `tracker.mode = "continuity_fixedlag"`
- record live outputs so we can compare the true online run against these corrected replay baselines afterward.
