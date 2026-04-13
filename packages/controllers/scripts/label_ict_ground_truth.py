#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from acies.controller.ict_ground_truth import SensorPoint, parse_gps_run_file, project_sensor_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate discrete ICT ground-truth labels for tracker development.")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/tkimura4/data/2024-03-29-ICT"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/design/artifacts/ict_tracker_2026-04-11/labels"))
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    parser.add_argument("--pair-gap-threshold-m", type=float, default=10.0)
    parser.add_argument("--direction-window", type=int, default=5, help="Rolling median window in samples for direction labeling.")
    parser.add_argument("--direction-threshold-mps", type=float, default=0.5)
    return parser.parse_args()


def load_sensor_locations(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "sensor_location.parquet").copy()
    out = df.rename(columns={"sensor_id": "node", "latitude": "sensor_latitude", "longitude": "sensor_longitude"}).copy()
    out["node"] = out["node"].astype(str)
    return out.sort_values("node").reset_index(drop=True)


def inventory_runs(data_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(data_dir.glob("run*_gps.parquet")):
        parsed = parse_gps_run_file(path)
        if parsed is None:
            continue
        rows.append({"run_id": parsed.run_id, "label": parsed.label, "gps_path": parsed.path})
    return sorted(rows, key=lambda row: int(row["run_id"]))


def load_run_meta(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "run_ids.parquet").sort_values("run_id").reset_index(drop=True)


def load_run_gps(gps_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(gps_path).copy()
    out = df.rename(columns={"time": "timestamp"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp", "latitude", "longitude"]).sort_values("timestamp").reset_index(drop=True)


def load_sensor_geometry(sensor_locations: pd.DataFrame, pair_gap_threshold_m: float) -> pd.DataFrame:
    points = [
        SensorPoint(node=row.node, latitude=float(row.sensor_latitude), longitude=float(row.sensor_longitude))
        for row in sensor_locations.itertuples(index=False)
    ]
    return pd.DataFrame(project_sensor_positions(points, pair_gap_threshold_m=pair_gap_threshold_m))


def load_run_dis_matrix(data_dir: Path, run_id: int, sensor_locations: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in sensor_locations.itertuples(index=False):
        path = data_dir / f"run{run_id}_{row.node}_dis.parquet"
        df = pd.read_parquet(path).copy()
        out = df.rename(columns={"time": "timestamp", "distance": row.node}).copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out[row.node] = pd.to_numeric(out[row.node], errors="coerce")
        frames.append(out[["timestamp", row.node]].dropna(subset=["timestamp"]))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")
    return merged.sort_values("timestamp").reset_index(drop=True)


def compute_station_trace(gps_df: pd.DataFrame, sensor_geometry: pd.DataFrame) -> pd.DataFrame:
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])
    center_x = float(sensor_geometry["x_m"].mean())
    center_y = float(sensor_geometry["y_m"].mean())
    stations = sensor_geometry.sort_values("station_axis_m")
    axis_dx = float(stations["station_x_m"].iloc[-1] - stations["station_x_m"].iloc[0])
    axis_dy = float(stations["station_y_m"].iloc[-1] - stations["station_y_m"].iloc[0])
    axis_norm = (axis_dx**2 + axis_dy**2) ** 0.5
    axis_x = axis_dx / axis_norm
    axis_y = axis_dy / axis_norm
    cos_ref = math.cos(ref_lat * (math.pi / 180.0))

    x_m = (gps_df["longitude"] - ref_lon) * (math.pi / 180.0) * 6_371_000.0 * cos_ref
    y_m = (gps_df["latitude"] - ref_lat) * (math.pi / 180.0) * 6_371_000.0
    axis_m = (x_m - center_x) * axis_x + (y_m - center_y) * axis_y

    out = gps_df[["timestamp", "latitude", "longitude"]].copy()
    out["axis_m"] = axis_m
    out["elapsed_s"] = (out["timestamp"] - out["timestamp"].iloc[0]).dt.total_seconds()
    out["axis_velocity_mps"] = out["axis_m"].diff().fillna(0.0) / out["elapsed_s"].diff().replace(0.0, pd.NA)
    out["axis_velocity_mps"] = out["axis_velocity_mps"].fillna(0.0)
    return out


def assign_labels(
    run_id: int,
    run_label: str,
    gps_df: pd.DataFrame,
    dis_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    direction_window: int,
    direction_threshold_mps: float,
) -> pd.DataFrame:
    merged = gps_df.merge(dis_df, on="timestamp", how="inner")
    sensor_cols = [row.node for row in sensor_geometry.sort_values("axis_m").itertuples(index=False)]
    station_map = sensor_geometry.set_index("node")["station_id"].to_dict()
    side_map = sensor_geometry.set_index("node")["cross_m"].to_dict()
    station_axis_map = sensor_geometry.groupby("station_id")["station_axis_m"].first().to_dict()

    records: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        distances = {sensor: float(getattr(row, sensor)) for sensor in sensor_cols}
        ranked_sensors = sorted(distances.items(), key=lambda item: item[1])
        best_sensor, best_sensor_distance = ranked_sensors[0]
        second_sensor_distance = ranked_sensors[1][1]

        station_distances: dict[int, float] = {}
        for sensor, distance in distances.items():
            station_id = int(station_map[sensor])
            best = station_distances.get(station_id)
            if best is None or distance < best:
                station_distances[station_id] = distance
        ranked_stations = sorted(station_distances.items(), key=lambda item: item[1])
        best_station, best_station_distance = ranked_stations[0]
        second_station_distance = ranked_stations[1][1]
        side = "positive_cross" if float(side_map[best_sensor]) >= 0.0 else "negative_cross"

        records.append(
            {
                "run_id": run_id,
                "label": run_label,
                "timestamp": row.timestamp,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "axis_m": float(row.axis_m),
                "elapsed_s": float(row.elapsed_s),
                "axis_velocity_mps": float(row.axis_velocity_mps),
                "nearest_sensor": best_sensor,
                "nearest_sensor_distance_m": best_sensor_distance,
                "second_sensor_distance_m": second_sensor_distance,
                "sensor_margin_m": second_sensor_distance - best_sensor_distance,
                "nearest_station": int(best_station),
                "nearest_station_distance_m": best_station_distance,
                "second_station_distance_m": second_station_distance,
                "station_margin_m": second_station_distance - best_station_distance,
                "nearest_station_axis_m": float(station_axis_map[int(best_station)]),
                "side_label": side,
            }
        )

    out = pd.DataFrame(records)
    smooth_vel = out["axis_velocity_mps"].rolling(window=direction_window, center=True, min_periods=1).median()
    out["direction"] = "ambiguous"
    out.loc[smooth_vel >= direction_threshold_mps, "direction"] = "toward_S4"
    out.loc[smooth_vel <= -direction_threshold_mps, "direction"] = "toward_S1"
    out["lap_progress_stationized"] = out["nearest_station"] + (out["axis_m"] - out["nearest_station_axis_m"]) / 54.0
    return out


def save_run_labels(run_labels: pd.DataFrame, out_dir: Path) -> None:
    run_id = int(run_labels["run_id"].iloc[0])
    run_labels.to_csv(out_dir / f"run{run_id}_discrete_labels.csv", index=False)


def plot_run_labels(run_labels: pd.DataFrame, out_dir: Path) -> None:
    run_id = int(run_labels["run_id"].iloc[0])
    station_colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}
    direction_colors = {"toward_S4": "tab:green", "toward_S1": "tab:red", "ambiguous": "0.5"}

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for station_id, group in run_labels.groupby("nearest_station"):
        axes[0].scatter(group["elapsed_s"], group["axis_m"], s=10, color=station_colors[int(station_id)], label=f"S{station_id}")
    axes[0].set_ylabel("Axis position (m)")
    axes[0].set_title(f"run{run_id} discrete station labels")
    axes[0].legend(loc="best", ncol=4, fontsize=8)

    side_values = run_labels["side_label"].map({"negative_cross": -1, "positive_cross": 1})
    axes[1].scatter(run_labels["elapsed_s"], side_values, s=8, c=run_labels["nearest_station"].map(station_colors))
    axes[1].set_yticks([-1, 1], labels=["negative_cross", "positive_cross"])
    axes[1].set_ylabel("Side")

    for direction, group in run_labels.groupby("direction"):
        axes[2].scatter(group["elapsed_s"], group["station_margin_m"], s=8, color=direction_colors[direction], label=direction)
    axes[2].set_ylabel("Station margin (m)")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / f"run{run_id}_discrete_labels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_run_transition_summary(run_labels: pd.DataFrame, out_dir: Path) -> None:
    run_id = int(run_labels["run_id"].iloc[0])
    station_seq = run_labels["nearest_station"].astype(int)
    changes = station_seq[station_seq.ne(station_seq.shift())].reset_index(drop=True)
    transitions = (
        pd.DataFrame({"src": changes[:-1].to_numpy(), "dst": changes[1:].to_numpy()})
        .value_counts()
        .rename("count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    if transitions.empty:
        ax.text(0.5, 0.5, "No transitions", ha="center", va="center")
    else:
        labels = [f"S{int(row.src)}→S{int(row.dst)}" for row in transitions.itertuples(index=False)]
        ax.bar(labels, transitions["count"], color="tab:blue")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(f"run{run_id} station transitions")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"run{run_id}_station_transitions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(summary_rows: list[dict[str, object]], out_dir: Path) -> None:
    df = pd.DataFrame(summary_rows).sort_values("run_id").reset_index(drop=True)
    df.to_csv(out_dir / "discrete_label_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = load_run_meta(args.data_dir)
    sensor_locations = load_sensor_locations(args.data_dir)
    sensor_geometry = load_sensor_geometry(sensor_locations, pair_gap_threshold_m=args.pair_gap_threshold_m)
    sensor_geometry.to_csv(args.out_dir / "sensor_geometry.csv", index=False)

    runs = inventory_runs(args.data_dir)
    if args.runs:
        selected = set(args.runs)
        runs = [run for run in runs if int(run["run_id"]) in selected]

    summary_rows: list[dict[str, object]] = []
    for run in runs:
        run_id = int(run["run_id"])
        label_row = run_meta[run_meta["run_id"] == run_id]
        run_label = str(label_row.iloc[0]["label"]) if not label_row.empty else str(run.get("label") or f"run{run_id}")

        gps_df = load_run_gps(Path(run["gps_path"]))
        station_trace = compute_station_trace(gps_df, sensor_geometry)
        dis_df = load_run_dis_matrix(args.data_dir, run_id, sensor_locations)
        run_labels = assign_labels(
            run_id=run_id,
            run_label=run_label,
            gps_df=station_trace,
            dis_df=dis_df,
            sensor_geometry=sensor_geometry,
            direction_window=args.direction_window,
            direction_threshold_mps=args.direction_threshold_mps,
        )

        save_run_labels(run_labels, args.out_dir)
        plot_run_labels(run_labels, args.out_dir)
        plot_run_transition_summary(run_labels, args.out_dir)

        direction_counts = run_labels["direction"].value_counts().to_dict()
        station_counts = run_labels["nearest_station"].value_counts().sort_index().to_dict()
        summary_rows.append(
            {
                "run_id": run_id,
                "label": run_label,
                "n_samples": len(run_labels),
                "median_station_margin_m": float(run_labels["station_margin_m"].median()),
                "median_sensor_margin_m": float(run_labels["sensor_margin_m"].median()),
                "ambiguous_fraction": float((run_labels["direction"] == "ambiguous").mean()),
                "toward_s4_fraction": float((run_labels["direction"] == "toward_S4").mean()),
                "toward_s1_fraction": float((run_labels["direction"] == "toward_S1").mean()),
                "station_counts": station_counts,
                "direction_counts": direction_counts,
            }
        )

    write_summary(summary_rows, args.out_dir)


if __name__ == "__main__":
    main()
