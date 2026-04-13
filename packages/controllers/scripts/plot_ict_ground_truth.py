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
    parser = argparse.ArgumentParser(description="Plot ICT GPS traces and sensor geometry for tracker development.")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/tkimura4/data/2024-03-29-ICT"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/ict_ground_truth"))
    parser.add_argument("--runs", type=int, nargs="*", default=None, help="Optional run ids to render.")
    parser.add_argument(
        "--pair-gap-threshold-m",
        type=float,
        default=10.0,
        help="Gap threshold in projected metres used to split paired sensors into stations.",
    )
    return parser.parse_args()


def load_run_metadata(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "run_ids.parquet").copy()
    return df.sort_values("run_id").reset_index(drop=True)


def load_sensor_locations(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "sensor_location.parquet").copy()
    out = df.rename(columns={"sensor_id": "node", "latitude": "sensor_latitude", "longitude": "sensor_longitude"}).copy()
    out["node"] = out["node"].astype(str)
    return out.sort_values("node").reset_index(drop=True)


def inventory_runs(data_dir: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for path in sorted(data_dir.glob("run*_gps.parquet")):
        parsed = parse_gps_run_file(path)
        if parsed is None:
            continue
        runs.append({"run_id": parsed.run_id, "label": parsed.label, "gps_path": parsed.path})
    return sorted(runs, key=lambda row: int(row["run_id"]))


def load_run_gps(gps_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(gps_path).copy()
    out = df.rename(columns={"time": "timestamp"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp", "latitude", "longitude"]).sort_values("timestamp").reset_index(drop=True)


def load_dis_timeseries(data_dir: Path, run_id: int, node: str) -> pd.DataFrame | None:
    path = data_dir / f"run{run_id}_{node}_dis.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).copy()
    out = df.rename(columns={"time": "timestamp", "distance": "distance_m"}).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
    out = out.dropna(subset=["timestamp", "distance_m"]).sort_values("timestamp").reset_index(drop=True)
    return out if not out.empty else None


def gps_distance_series(gps_df: pd.DataFrame, sensor_lat: float, sensor_lon: float) -> pd.Series:
    radius_m = 6_371_000.0
    lat1 = pd.to_numeric(gps_df["latitude"], errors="coerce").astype(float)
    lon1 = pd.to_numeric(gps_df["longitude"], errors="coerce").astype(float)
    phi1 = lat1 * (math.pi / 180.0)
    phi2 = sensor_lat * (math.pi / 180.0)
    dphi = (sensor_lat - lat1) * (math.pi / 180.0)
    dlambda = (sensor_lon - lon1) * (math.pi / 180.0)
    a = (dphi / 2.0).map(lambda x: math.sin(float(x)) ** 2)
    a += phi1.map(math.cos) * math.cos(phi2) * dlambda.map(lambda x: math.sin(float(x) / 2.0) ** 2)
    return a.map(lambda x: 2.0 * radius_m * math.atan2(math.sqrt(float(x)), math.sqrt(1.0 - float(x))))


def build_sensor_geometry(sensor_locations: pd.DataFrame, pair_gap_threshold_m: float) -> pd.DataFrame:
    points = [
        SensorPoint(node=row.node, latitude=float(row.sensor_latitude), longitude=float(row.sensor_longitude))
        for row in sensor_locations.itertuples(index=False)
    ]
    return pd.DataFrame(project_sensor_positions(points, pair_gap_threshold_m=pair_gap_threshold_m))


def compute_station_trace(gps_df: pd.DataFrame, sensor_geometry: pd.DataFrame) -> pd.DataFrame:
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])
    center_x = float(sensor_geometry["x_m"].mean())
    center_y = float(sensor_geometry["y_m"].mean())
    station_a = sensor_geometry.sort_values("axis_m")
    axis_dx = float(station_a["station_x_m"].iloc[-1] - station_a["station_x_m"].iloc[0])
    axis_dy = float(station_a["station_y_m"].iloc[-1] - station_a["station_y_m"].iloc[0])
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
    return out


def save_sensor_geometry(sensor_geometry: pd.DataFrame, out_dir: Path) -> None:
    sensor_geometry.to_csv(out_dir / "sensor_geometry.csv", index=False)
    stations = (
        sensor_geometry.groupby("station_id", as_index=False)
        .agg(
            station_axis_m=("station_axis_m", "first"),
            station_cross_m=("station_cross_m", "first"),
            station_x_m=("station_x_m", "first"),
            station_y_m=("station_y_m", "first"),
            node_count=("node", "count"),
        )
        .sort_values("station_id")
    )
    stations.to_csv(out_dir / "station_geometry.csv", index=False)


def plot_overview(
    all_runs: list[dict[str, object]], run_meta: pd.DataFrame, sensor_locations: pd.DataFrame, sensor_geometry: pd.DataFrame, out_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(sensor_locations["sensor_longitude"], sensor_locations["sensor_latitude"], c="black", s=60, label="sensors")
    for row in sensor_locations.itertuples(index=False):
        ax.text(row.sensor_longitude, row.sensor_latitude, row.node, fontsize=9, ha="left", va="bottom")

    stations = sensor_geometry.groupby("station_id", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    )
    ax.scatter(stations["longitude"], stations["latitude"], c="tab:red", s=80, marker="s", label="stations")
    for row in stations.itertuples(index=False):
        ax.text(row.longitude, row.latitude, f"S{row.station_id}", fontsize=9, ha="right", va="top", color="tab:red")

    cmap = plt.get_cmap("tab10")
    for idx, run in enumerate(all_runs):
        gps_df = load_run_gps(Path(run["gps_path"]))
        run_id = int(run["run_id"])
        label_row = run_meta[run_meta["run_id"] == run_id]
        label = label_row.iloc[0]["label"] if not label_row.empty and "label" in label_row.columns else run.get("label") or f"run{run_id}"
        color = cmap(idx % 10)
        ax.plot(gps_df["longitude"], gps_df["latitude"], color=color, alpha=0.55, linewidth=1.5, label=f"run{run_id}:{label}")
        ax.scatter(gps_df["longitude"].iloc[0], gps_df["latitude"].iloc[0], color=color, marker="^", s=40)
        ax.scatter(gps_df["longitude"].iloc[-1], gps_df["latitude"].iloc[-1], color=color, marker="v", s=40)

    ax.set_title("ICT sensor layout and GPS traces")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "overview_all_runs_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_run_map(run_id: int, gps_df: pd.DataFrame, sensor_locations: pd.DataFrame, sensor_geometry: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(gps_df["longitude"], gps_df["latitude"], c=range(len(gps_df)), cmap="viridis", s=10)
    fig.colorbar(scatter, ax=ax, label="sample index")
    ax.plot(gps_df["longitude"], gps_df["latitude"], color="0.7", linewidth=1.0, alpha=0.6)

    ax.scatter(sensor_locations["sensor_longitude"], sensor_locations["sensor_latitude"], c="black", s=60, label="sensors")
    for row in sensor_locations.itertuples(index=False):
        ax.text(row.sensor_longitude, row.sensor_latitude, row.node, fontsize=9, ha="left", va="bottom")

    stations = sensor_geometry.groupby("station_id", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    )
    ax.scatter(stations["longitude"], stations["latitude"], c="tab:red", s=80, marker="s", label="stations")
    ax.scatter(gps_df["longitude"].iloc[0], gps_df["latitude"].iloc[0], c="tab:green", marker="^", s=70, label="start")
    ax.scatter(gps_df["longitude"].iloc[-1], gps_df["latitude"].iloc[-1], c="tab:orange", marker="v", s=70, label="end")

    ax.set_title(f"run{run_id} GPS trace and sensors")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"run{run_id}_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_run_progress(run_id: int, station_trace: pd.DataFrame, sensor_geometry: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(station_trace["elapsed_s"], station_trace["axis_m"], color="tab:blue", linewidth=1.5)

    for row in (
        sensor_geometry.groupby("station_id", as_index=False)
        .agg(station_axis_m=("station_axis_m", "first"))
        .sort_values("station_id")
        .itertuples(index=False)
    ):
        ax.axhline(row.station_axis_m, color="tab:red", linestyle="--", alpha=0.4)
        ax.text(station_trace["elapsed_s"].max(), row.station_axis_m, f"S{row.station_id}", color="tab:red", ha="right", va="bottom")

    ax.set_title(f"run{run_id} projected station-axis progress")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Projected along-road position (m)")
    fig.tight_layout()
    fig.savefig(out_dir / f"run{run_id}_progress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sensor_distances(run_id: int, gps_df: pd.DataFrame, sensor_locations: pd.DataFrame, data_dir: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    have_any = False

    for row in sensor_locations.itertuples(index=False):
        dis_df = load_dis_timeseries(data_dir, run_id, row.node)
        if dis_df is not None:
            plot_df = dis_df.copy()
            plot_df["elapsed_s"] = (plot_df["timestamp"] - plot_df["timestamp"].iloc[0]).dt.total_seconds()
            ax.plot(plot_df["elapsed_s"], plot_df["distance_m"], linewidth=1.2, label=f"{row.node} dis")
            have_any = True
            continue

        plot_df = gps_df[["timestamp"]].copy()
        plot_df["elapsed_s"] = (gps_df["timestamp"] - gps_df["timestamp"].iloc[0]).dt.total_seconds()
        plot_df["distance_m"] = gps_distance_series(gps_df, float(row.sensor_latitude), float(row.sensor_longitude))
        ax.plot(plot_df["elapsed_s"], plot_df["distance_m"], linewidth=1.0, linestyle="--", label=f"{row.node} gps")
        have_any = True

    if not have_any:
        plt.close(fig)
        return

    ax.set_title(f"run{run_id} distance to each sensor")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Distance (m)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"run{run_id}_sensor_distance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = load_run_metadata(args.data_dir)
    sensor_locations = load_sensor_locations(args.data_dir)
    sensor_geometry = build_sensor_geometry(sensor_locations, pair_gap_threshold_m=args.pair_gap_threshold_m)
    save_sensor_geometry(sensor_geometry, args.out_dir)

    runs = inventory_runs(args.data_dir)
    if args.runs:
        selected = set(args.runs)
        runs = [run for run in runs if int(run["run_id"]) in selected]

    plot_overview(runs, run_meta, sensor_locations, sensor_geometry, args.out_dir)

    for run in runs:
        run_id = int(run["run_id"])
        gps_df = load_run_gps(Path(run["gps_path"]))
        station_trace = compute_station_trace(gps_df, sensor_geometry)
        plot_run_map(run_id, gps_df, sensor_locations, sensor_geometry, args.out_dir)
        plot_run_progress(run_id, station_trace, sensor_geometry, args.out_dir)
        plot_sensor_distances(run_id, gps_df, sensor_locations, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
