# West Point Line Topology: Dynamic Road Polyline Tuning
**Date:** 2026-04-15

## Overview
This document captures the design changes and evaluation results for eliminating topology "jumpbacks" when adapting the West Point sensor line topology to the physical road coordinates.

The previous strategy implicitly allocated exactly 5 points for every sensor. Because sensor spacing ranges from ~13m (`rs5` to `rs7`) to ~47m (`rs2` to `rs7`), forcing an equal number of points per segment caused spatial bunching and overlapping (jumpbacks) along the target path.

## Modifications

1. **Exact Road Alignment**: The road segment explicitly defined by coordinates tracing the sensor-side road was extracted to function as a `road_polyline`. The user provided the exact ground-truth road segment:
   - 41.35376626231936, -74.05516544352628 (Start)
   - 41.354321598961405, -74.05467053194602
   - 41.35475437526667, -74.05378275241027
   - 41.35475054540048, -74.05292048378075 (End)
   The total 1D sequence length computes to exactly 234.6m using an equirectangular coordinate approximation based on a central reference point.

2. **Boundary Segmentation**: Sensors were mapped and projected locally via the Cartesian XY system onto the 1D polyline. The distances along the path for each sensor were computed:
   - `rs1`: 12.0m
   - `rs3`: 47.8m
   - `rs5`: 56.6m
   - `rs7`: 73.6m
   - `rs2`: 120.8m
   - `rs6`: 167.7m
   - `rs8`: 183.8m
   - `rs10`: 209.5m

   To prevent overlaps and jumpbacks, the path was strictly segmented by taking the midpoint between adjacent sensors along the road. The final boundaries for point allocation are:
   - `rs1`: Interval `[0.0m, 29.9m]`
   - `rs3`: Interval `[29.9m, 52.2m]`
   - `rs5`: Interval `[52.2m, 65.1m]`
   - `rs7`: Interval `[65.1m, 97.2m]`
   - `rs2`: Interval `[97.2m, 144.3m]`
   - `rs6`: Interval `[144.3m, 175.7m]`
   - `rs8`: Interval `[175.7m, 196.7m]`
   - `rs10`: Interval `[196.7m, 234.6m]`

3. **Dynamic Point Allocation**: Rather than maintaining a rigid 5 points per sensor, the new configuration (`ContinuityLatticeConfig`) calculates the segment length, divides it by `target_point_spacing_m = 6.0`, and dynamically allocates points. This resolves the spatial bunching (e.g. `rs5` interval gets fewer points, `rs2` interval gets more points) and places them perfectly along the path tangent without any structural inversions.
## Results

Because the point distribution precisely mimics the road path, there are no longer any internal loop inversions (jumpbacks).

### Runtime Trace Plots (Full-fit)
Here are the visualization traces from the runtime controller across a comprehensive set of runs. Points are plotted at a 5-second interval:

**Run 0:**
![Run 0](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run0_controller_runtime_clean_xy.png)

**Run 1:**
![Run 1](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run1_controller_runtime_clean_xy.png)

**Run 2:**
![Run 2](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run2_controller_runtime_clean_xy.png)

**Run 3:**
![Run 3](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run3_controller_runtime_clean_xy.png)

**Run 4:**
![Run 4](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run4_controller_runtime_clean_xy.png)

**Run 5:**
![Run 5](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run5_controller_runtime_clean_xy.png)

**Run 6:**
![Run 6](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run6_controller_runtime_clean_xy.png)

**Run 7:**
![Run 7](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run7_controller_runtime_clean_xy.png)

**Run 8:**
![Run 8](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run8_controller_runtime_clean_xy.png)

**Run 9:**
![Run 9](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run9_controller_runtime_clean_xy.png)

**Run 10:**
![Run 10](../artifacts/west_point_2026-04-14/runtime_eval_corridor25_fullfit/run10_controller_runtime_clean_xy.png)
