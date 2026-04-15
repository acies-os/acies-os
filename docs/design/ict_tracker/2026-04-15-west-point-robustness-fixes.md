# West Point Line Topology: Robustness Fixes
**Date:** 2026-04-15

## Overview
This document outlines the evaluation results for fixing "jumpy" and "tangled" trace artifacts observed around dense sensor segments (particularly `rs2`, `rs6`, and `rs8`) following the deployment of the dynamic road polyline.

## Root Cause

In dense segments, every lattice node belonging to the same sensor receives an identical anchor score from `_anchor_score_vector` — there is no spatial gradient within the segment. With the original `stay_penalty = 0.05` versus a net forward hop cost of `0.35` (hop_penalty 0.7 − direction_bonus 0.35), staying still was **7× cheaper** than advancing. In flat-emission regions the Viterbi path therefore refused to move, getting stranded at the entry edge of a segment until the next sensor fired and forced an abrupt jump.

## Modifications

1. **Reverted Continuous Anchor Score Gradient**: An earlier attempt fixed stranding by penalizing the Viterbi spatial index to pull the path toward the physical center of each segment (`-0.02 * abs(node.point_index - center_point_index)`). This was a regression: it created "center-snapping" artifacts and dropped accuracy from 50.0% to 49.67% while increasing Mean XY Error from 34.01m to 34.09m. This logic has been removed.

2. **Direction-Conditional Stay Penalty** (`FixedLagContinuityRuntime._transition_score`): When the direction signal is active (`expected_sign != 0`), the stay cost is raised to 85% of the net forward hop cost (0.2975 with default parameters), reducing the stay-vs-hop ratio from 7× to ~1.17×. When direction is ambiguous (`expected_sign == 0`) the baseline `stay_penalty = 0.05` is preserved, avoiding artificial movement pressure during phases where there is no valid directional evidence.

   ```python
   if delta == 0:
       if expected_sign != 0:
           net_forward_cost = hop_penalty - direction_bonus  # 0.35
           return -(net_forward_cost * 0.85)               # 0.2975
       return -self.assets.config.stay_penalty             # 0.05
   ```

3. **Evaluation Script Fix**: Patched a bug in `eval_ict_continuity_layer.py` (`_estimate_delay_seconds`) where bounds checking failed when the time lag exceeded the array size, allowing successful end-to-end evaluation.

## Results

| Metric | Baseline | Fixed |
|--------|----------|-------|
| Joint Station/Side Accuracy | 50.00% | **50.76%** |
| Mean XY Error | 34.01m | **33.49m** |
| Point Accuracy | 18.40% | **19.19%** |
| P95 XY Error | 146.6m | **145.7m** |
| n_samples | 3674 | 3674 |

Evaluation command:
```bash
python packages/controllers/scripts/eval_tracker_runtime_replay.py \
    --asset-dir docs/design/artifacts/west_point_2026-04-14/assets_corridor25 \
    --cache-dir docs/design/artifacts/west_point_2026-04-14/cache_wp \
    --labels-dir docs/design/artifacts/west_point_2026-04-14/labels_wp \
    --max-nearest-sensor-distance-m 25 \
    --out-dir docs/design/artifacts/west_point_2026-04-14/runtime_eval_fixes_wp
```

### Runtime Trace Plots (Fixed)
Visualization traces from the updated runtime controller across the evaluated runs, separated into Pass 0 and Pass 1.

**Run 2:**
![Run 2 Pass 0 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run2_pass0_controller_runtime_clean_xy.png)
![Run 2 Pass 0 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run2_pass0_controller_runtime_timing.png)
![Run 2 Pass 1 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run2_pass1_controller_runtime_clean_xy.png)
![Run 2 Pass 1 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run2_pass1_controller_runtime_timing.png)

**Run 3:**
![Run 3 Pass 0 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run3_pass0_controller_runtime_clean_xy.png)
![Run 3 Pass 0 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run3_pass0_controller_runtime_timing.png)
![Run 3 Pass 1 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run3_pass1_controller_runtime_clean_xy.png)
![Run 3 Pass 1 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run3_pass1_controller_runtime_timing.png)

**Run 6:**
![Run 6 Pass 0 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run6_pass0_controller_runtime_clean_xy.png)
![Run 6 Pass 0 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run6_pass0_controller_runtime_timing.png)
![Run 6 Pass 1 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run6_pass1_controller_runtime_clean_xy.png)
![Run 6 Pass 1 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run6_pass1_controller_runtime_timing.png)

**Run 7:**
![Run 7 Pass 0 XY](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run7_pass0_controller_runtime_clean_xy.png)
![Run 7 Pass 0 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run7_pass0_controller_runtime_timing.png)
![Run 7 Pass 1 XY](../artifacts/west_point_2026-04-14/rok,untime_eval_fixes_wp/run7_pass1_controller_runtime_clean_xy.png)
![Run 7 Pass 1 Timing](../artifacts/west_point_2026-04-14/runtime_eval_fixes_wp/run7_pass1_controller_runtime_timing.png)
