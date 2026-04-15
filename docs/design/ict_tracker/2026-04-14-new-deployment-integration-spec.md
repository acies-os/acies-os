# New Deployment Integration Specification

Date: `2026-04-14`

## 1. Goal

Provide a concrete guide and specification for translating a new, open-road/line sensor deployment into the `DeploymentAssets` format required by the continuity tracker, without rewriting the controller runtime.

## 2. Sensor Geometry Translation

When integrating a new deployment, the first step is mapping physical sensor coordinates into the tracker's required logical coordinate system.

**Expected Inputs:**
- A table or list containing `sensor_id`, `latitude`, and `longitude`.

**Translation Process:**
1. **Establish the Reference Frame:** Unlike the ICT loop where the origin is arbitrary, for a line topology, select the "start" or "entrance" of the monitored area as the origin (0, 0).
2. **Determine `station_id`:** Map each sensor's longitudinal progress along the linear road to a discrete `station_id`. Stations should increment sequentially from the entrance to the exit.
3. **Determine `side_label`:** Map the lateral placement of the sensor relative to the road centerline. For example, sensors on the left side of the road direction of travel could be `negative_cross`, and right side `positive_cross`.

## 3. Lattice Point Generation (Loop -> Line Topology)

The continuity tracker relies on a discrete state graph defined in `lattice_points.csv`.

**Structure Requirements:**
- The lattice must define discrete `loop_node_index` values sequentially from the start of the line to the end.
- Each node must map back to its nearest `station_id`, `side_label`, and corresponding `sensor_node`.

**Boundary Handling:**
- In a loop topology, the lattice wraps from the highest index back to 0. 
- In a line topology, this wrapping must be removed. The state graph represents a bounded line, and the edges must clamp at the entrance and exit boundaries.

**Configuration Update:**
- The `topology` flag inside the generated `metadata.json` must be explicitly set to `"line"` instead of `"loop"`. The `FixedLagContinuityRuntime` already uses this configuration flag to properly handle boundary math during the Viterbi decoding step.

## 4. Modifying the Asset Builder

To support generating these assets programmatically, the existing `build_ict_tracker_assets.py` script needs to be extended.

**Targeted Changes:**
1. **CLI Arguments:** Add a `--topology` argument (choices: `["loop", "line"]`, default: `"loop"`) to control the generation mode.
2. **Line Lattice Generation:** Implement a `build_line_lattice()` function as an alternative to the existing `build_loop_lattice()`. This function will generate the nodes without applying the circular wrapping logic.
3. **Edge Connections:** Ensure the `lattice_edges.csv` generation logic respects the topology. For `"line"`, the script must not generate an edge connecting the final node back to the first node.

## 5. Re-tuning the Pooled Median Templates

The observation model driving the tracker is the `pooled_median` energy template. The beauty of this integration path is that the observation algorithm itself requires zero changes.

**Retuning Process:**
1. Align the new, unseen sensor data with the newly generated line lattice (using GPS ground truth if available, or estimated state mappings).
2. Aggregate the normalized features grouped by `gt_loop_node_index`.
3. Compute the `median` for each feature column at each node.
4. Output this data to `continuity_templates.csv`.

Once the `metadata.json`, `lattice_points.csv`, and updated templates are bundled in the `asset_dir`, the controller can be pointed to it and will immediately track vehicles along the new linear deployment.
