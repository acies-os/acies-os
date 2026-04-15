import sys
import pandas as pd
import numpy as np
sys.path.insert(0, 'packages/controllers/src')
sys.path.insert(0, 'packages/controllers/scripts')
from eval_ict_simple_tracker import inventory_signal_files, compute_feature_file, maybe_load_or_compute, build_aligned_dataset, load_all_labels, load_sensor_locations, build_sensor_geometry, apply_corridor_filter
from pathlib import Path
from build_ict_tracker_assets import _feature_columns, _normalize_with_global_stats, ContinuityLatticeConfig, build_point_lattice, build_lattice_edges, assign_ground_truth_loop_nodes, _node_feature_stat

DATA_DIR=Path("/home/jinyang7/data/2026-04-14-West-Point")
ARTIFACTS_DIR=Path("docs/design/artifacts/west_point_2026-04-14")

print("Loading labels...")
labels_df = load_all_labels(ARTIFACTS_DIR / "labels", runs=None)
print("Loaded labels.")

sensor_locations = load_sensor_locations(DATA_DIR)
sensor_geometry = build_sensor_geometry(sensor_locations)
print("Loaded sensor geometry.")

print("Loading features...")
features_df = pd.read_parquet(ARTIFACTS_DIR / "cache" / "signal_features_w1_s1.parquet")
print("Loaded features.")

print("Building aligned dataset...")
aligned_df_raw = build_aligned_dataset(labels_df, features_df, runs=None)
print("Built aligned dataset.")

print("Applying corridor filter...")
aligned_df_raw = apply_corridor_filter(aligned_df_raw, 25.0)

feature_cols = _feature_columns(aligned_df_raw, modality="mic")
normalized_df, feature_stats = _normalize_with_global_stats(aligned_df_raw, feature_cols=feature_cols)

print("Building lattice...")
cfg = ContinuityLatticeConfig(point_count=5, center_quantile=0.15)
lattice_df = build_point_lattice(normalized_df, sensor_geometry=sensor_geometry, config=cfg, topology="line")
print("Built lattice.")

print("Assigning ground truth...")
labeled_df = assign_ground_truth_loop_nodes(normalized_df, lattice_df)
print("Assigned ground truth.")

