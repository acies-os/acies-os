#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ICT one-loop ground-truth and tracker traces.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("docs/design/artifacts/ict_tracker_2026-04-11/simple_tracker/tracker_predictions.csv"),
    )
    parser.add_argument(
        "--sensor-geometry",
        type=Path,
        default=Path("docs/design/artifacts/ict_tracker_2026-04-11/plots/sensor_geometry.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/design/artifacts/ict_tracker_2026-04-12/loop_plots"),
    )
    parser.add_argument("--tracker-mode", default="hybrid_mic")
    parser.add_argument("--runs", type=int, nargs="*", default=[0, 1, 3])
    return parser.parse_args()


def latlon_to_xy_m(lat: pd.Series, lon: pd.Series, ref_lat: float, ref_lon: float) -> tuple[pd.Series, pd.Series]:
    deg = math.pi / 180.0
    earth_r = 6_371_000.0
    cos_ref = math.cos(ref_lat * deg)
    x = (lon - ref_lon) * deg * earth_r * cos_ref
    y = (lat - ref_lat) * deg * earth_r
    return x, y


def find_loop_bounds(axis_m: pd.Series) -> tuple[int, int]:
    smooth = axis_m.rolling(window=5, center=True, min_periods=1).mean().reset_index(drop=True)
    values = smooth.to_list()
    minima = []
    for i in range(10, len(values) - 10):
        window = values[i - 10 : i + 11]
        if values[i] == min(window) and values[i] < -85:
            if not minima or i - minima[-1] > 20:
                minima.append(i)
    for a, b in zip(minima[:-1], minima[1:]):
        if max(values[a:b+1]) > 85:
            return a, b
    raise ValueError("Could not find a full loop segment")


def map_predicted_state_to_xy(pred_df: pd.DataFrame, sensor_geometry: pd.DataFrame) -> pd.DataFrame:
    sensor_side = {}
    for station_id, group in sensor_geometry.groupby("station_id"):
        pos = group.sort_values("cross_m").iloc[-1]
        neg = group.sort_values("cross_m").iloc[0]
        sensor_side[(int(station_id), "positive_cross")] = (float(pos["latitude"]), float(pos["longitude"]))
        sensor_side[(int(station_id), "negative_cross")] = (float(neg["latitude"]), float(neg["longitude"]))

    pred_lats = []
    pred_lons = []
    for row in pred_df.itertuples(index=False):
        pred_lat, pred_lon = sensor_side[(int(row.pred_station), str(row.pred_side))]
        pred_lats.append(pred_lat)
        pred_lons.append(pred_lon)
    pred_df = pred_df.copy()
    pred_df["pred_latitude"] = pred_lats
    pred_df["pred_longitude"] = pred_lons
    return pred_df


def plot_loop(run_df: pd.DataFrame, sensor_geometry: pd.DataFrame, out_path: Path) -> None:
    ref_lat = float(sensor_geometry["latitude"].mean())
    ref_lon = float(sensor_geometry["longitude"].mean())
    gt_x, gt_y = latlon_to_xy_m(run_df["latitude"], run_df["longitude"], ref_lat, ref_lon)
    pred_x, pred_y = latlon_to_xy_m(run_df["pred_latitude"], run_df["pred_longitude"], ref_lat, ref_lon)
    sensor_x, sensor_y = latlon_to_xy_m(sensor_geometry["latitude"], sensor_geometry["longitude"], ref_lat, ref_lon)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_x, gt_y, color="tab:blue", linewidth=2, label="ground truth GPS")
    ax.plot(pred_x, pred_y, color="tab:orange", linewidth=1.5, alpha=0.9, label="hybrid_mic prediction")
    ax.scatter(gt_x.iloc[0], gt_y.iloc[0], color="tab:blue", marker="o", s=60, label="GT start")
    ax.scatter(gt_x.iloc[-1], gt_y.iloc[-1], color="tab:blue", marker="s", s=60, label="GT end")
    ax.scatter(pred_x.iloc[0], pred_y.iloc[0], color="tab:orange", marker="o", s=50, label="Pred start")
    ax.scatter(pred_x.iloc[-1], pred_y.iloc[-1], color="tab:orange", marker="s", s=50, label="Pred end")
    ax.scatter(sensor_x, sensor_y, color="black", s=70, label="sensors")

    for row, sx, sy in zip(sensor_geometry.itertuples(index=False), sensor_x, sensor_y):
        ax.text(sx, sy, row.node, fontsize=9, ha="left", va="bottom")

    run_id = int(run_df["run_id"].iloc[0])
    label = str(run_df["label"].iloc[0])
    t0 = float(run_df["elapsed_s"].iloc[0])
    t1 = float(run_df["elapsed_s"].iloc[-1])
    ax.set_title(f"run{run_id} {label}: one-loop GT vs hybrid_mic ({t0:.0f}s to {t1:.0f}s)")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.predictions)
    pred = pred[pred["tracker_mode"] == args.tracker_mode].copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    sensor_geometry = pd.read_csv(args.sensor_geometry)
    pred = map_predicted_state_to_xy(pred, sensor_geometry)

    loop_rows = []
    for run_id in args.runs:
        run_df = pred[pred["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
        if run_df.empty:
            continue
        start_idx, end_idx = find_loop_bounds(run_df["axis_m"])
        loop_df = run_df.iloc[start_idx : end_idx + 1].copy()
        out_path = args.out_dir / f"run{run_id}_{args.tracker_mode}_one_loop_xy.png"
        plot_loop(loop_df, sensor_geometry, out_path)
        loop_rows.append(
            {
                "run_id": run_id,
                "label": str(loop_df["label"].iloc[0]),
                "start_elapsed_s": float(loop_df["elapsed_s"].iloc[0]),
                "end_elapsed_s": float(loop_df["elapsed_s"].iloc[-1]),
                "n_points": len(loop_df),
                "path": out_path.name,
            }
        )

    pd.DataFrame(loop_rows).to_csv(args.out_dir / "loop_segments.csv", index=False)


if __name__ == "__main__":
    main()
