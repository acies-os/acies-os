#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from acies.controller.ict_continuity import (
    ContinuityLatticeConfig,
    assign_ground_truth_loop_nodes,
    build_lattice_edges,
    build_point_lattice,
    infer_loop_sensor_order,
)

from eval_ict_simple_tracker import (
    build_aligned_dataset,
    build_sensor_geometry,
    compute_feature_file,
    inventory_signal_files,
    load_all_labels,
    load_sensor_locations,
    maybe_load_or_compute,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deployment assets for the controller ICT continuity runtime.")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/tkimura4/data/2024-03-29-ICT"))
    parser.add_argument("--labels-dir", type=Path, default=Path("docs/design/artifacts/ict_tracker_2026-04-11/labels"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("docs/design/artifacts/ict_tracker_2026-04-14_runtime_integration/cache"))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    parser.add_argument("--point-count", type=int, default=5)
    parser.add_argument("--center-quantile", type=float, default=0.15)
    parser.add_argument("--topology", choices=["loop", "line"], default="loop")
    parser.add_argument("--modality", default="mic")
    parser.add_argument("--lag-steps", type=int, default=5)
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame, modality: str) -> list[str]:
    return sorted([col for col in df.columns if col.endswith(f"__{modality}")])


def _normalize_with_global_stats(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean": [float(df[col].mean()) for col in feature_cols],
            "std": [max(float(df[col].std(ddof=0)), 1e-9) for col in feature_cols],
        }
    )
    out = df.copy()
    for row in stats.itertuples(index=False):
        out[row.feature] = (out[row.feature] - float(row.mean)) / float(row.std)
    return out, stats


def _node_feature_stat(train_group: pd.DataFrame, lattice_df: pd.DataFrame, feature_cols: list[str], stat: str) -> pd.DataFrame:
    template_cols = ["gt_loop_node_index"] + feature_cols
    grouped = train_group[template_cols].groupby("gt_loop_node_index")[feature_cols]
    if stat == "mean":
        node_df = grouped.mean()
    elif stat == "median":
        node_df = grouped.median()
    else:
        raise ValueError(stat)
    return node_df.reindex(lattice_df["loop_node_index"]).interpolate(limit_direction="both").fillna(0.0)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = set(args.runs) if args.runs else None
    labels_df = load_all_labels(args.labels_dir, runs=list(selected_runs) if selected_runs else None)
    sensor_locations = load_sensor_locations(args.data_dir)
    sensor_geometry = build_sensor_geometry(sensor_locations)
    sensor_geometry.to_csv(args.out_dir / "sensor_geometry.csv", index=False)

    items = inventory_signal_files(args.data_dir)
    if selected_runs is not None:
        items = [item for item in items if item.run in selected_runs]

    features_df = maybe_load_or_compute(
        args.cache_dir / f"signal_features_w{args.window_seconds:g}_s{args.stride_seconds:g}.parquet",
        lambda: pd.concat(
            [compute_feature_file(item, window_seconds=args.window_seconds, stride_seconds=args.stride_seconds) for item in items],
            ignore_index=True,
        ),
    )
    aligned_df_raw = maybe_load_or_compute(
        args.cache_dir / "aligned_energy_labels.parquet",
        lambda: build_aligned_dataset(labels_df, features_df, runs=list(selected_runs) if selected_runs else None),
    )

    feature_cols = _feature_columns(aligned_df_raw, modality=args.modality)
    if not feature_cols:
        raise ValueError(f"No feature columns found for modality={args.modality}")

    normalized_df, feature_stats = _normalize_with_global_stats(aligned_df_raw, feature_cols=feature_cols)
    feature_stats.to_csv(args.out_dir / "feature_stats.csv", index=False)

    cfg = ContinuityLatticeConfig(point_count=args.point_count, center_quantile=args.center_quantile)
    lattice_df = build_point_lattice(normalized_df, sensor_geometry=sensor_geometry, config=cfg)
    lattice_df.to_csv(args.out_dir / "lattice_points.csv", index=False)
    build_lattice_edges(lattice_df).to_csv(args.out_dir / "lattice_edges.csv", index=False)

    labeled_df = assign_ground_truth_loop_nodes(normalized_df, lattice_df)
    labeled_df.to_parquet(args.out_dir / "normalized_training_rows.parquet", index=False)

    station_side_templates = (
        labeled_df.groupby(["nearest_station", "side_label"])[feature_cols]
        .mean()
        .reset_index()
        .rename(columns={"nearest_station": "station_id"})
    )
    station_side_templates.to_csv(args.out_dir / "station_side_templates.csv", index=False)

    continuity_templates = _node_feature_stat(
        labeled_df,
        lattice_df=lattice_df,
        feature_cols=feature_cols,
        stat="median",
    ).reset_index().rename(columns={"index": "loop_node_index"})
    continuity_templates.to_csv(args.out_dir / "continuity_templates.csv", index=False)

    metadata = {
        "deployment_name": args.out_dir.name,
        "topology": args.topology,
        "modality": args.modality,
        "feature_names": feature_cols,
        "sensor_order": infer_loop_sensor_order(sensor_geometry),
        "runtime": {
            "lag_steps": args.lag_steps,
            "template_weight": 1.8,
            "anchor_weight": 1.0,
            "neighbor_anchor_weight": 0.55,
            "max_step_nodes": 2,
            "hop_penalty": 0.7,
            "direction_bonus": 0.35,
            "direction_penalty": 0.25,
            "stay_penalty": 0.05,
            "reset_penalty": 4.0,
            "margin_scale": 1.0,
            "change_margin_threshold": 0.75,
            "hybrid_margin_threshold": 0.70,
            "direction_window": 5,
            "direction_threshold": 0.08,
            "anchor_temperature": 1.25,
            "lock_run_direction": False,
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
