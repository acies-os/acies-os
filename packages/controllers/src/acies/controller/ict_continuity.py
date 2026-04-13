from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


_DEG_TO_RAD = math.pi / 180.0
_EARTH_R = 6_371_000.0


@dataclass(frozen=True)
class ContinuityLatticeConfig:
    point_count: int = 5
    center_quantile: float = 0.15
    fallback_half_span_m: float = 10.0
    max_half_span_m: float = 25.0


def latlon_to_xy_m(lat: pd.Series, lon: pd.Series, ref_lat: float, ref_lon: float) -> tuple[pd.Series, pd.Series]:
    cos_ref = math.cos(ref_lat * _DEG_TO_RAD)
    x = (lon - ref_lon) * _DEG_TO_RAD * _EARTH_R * cos_ref
    y = (lat - ref_lat) * _DEG_TO_RAD * _EARTH_R
    return x, y


def xy_to_latlon_m(x_m: np.ndarray, y_m: np.ndarray, ref_lat: float, ref_lon: float) -> tuple[np.ndarray, np.ndarray]:
    cos_ref = math.cos(ref_lat * _DEG_TO_RAD)
    lon = ref_lon + (x_m / (_EARTH_R * cos_ref)) / _DEG_TO_RAD
    lat = ref_lat + (y_m / _EARTH_R) / _DEG_TO_RAD
    return lat, lon


def infer_loop_sensor_order(sensor_geometry: pd.DataFrame) -> list[str]:
    negative = []
    positive = []
    for station_id, group in sensor_geometry.sort_values("station_id").groupby("station_id"):
        ordered = group.sort_values("cross_m").reset_index(drop=True)
        negative.append(str(ordered.iloc[0]["node"]))
        positive.append(str(ordered.iloc[-1]["node"]))
    return negative + list(reversed(positive))


def sensor_for_station_side(sensor_geometry: pd.DataFrame, station_id: int, side_label: str) -> str:
    group = sensor_geometry[sensor_geometry["station_id"] == station_id].sort_values("cross_m").reset_index(drop=True)
    if group.empty:
        raise KeyError(f"No sensor geometry for station {station_id}")
    if side_label == "negative_cross":
        return str(group.iloc[0]["node"])
    if side_label == "positive_cross":
        return str(group.iloc[-1]["node"])
    raise KeyError(f"Unknown side label {side_label}")


def _split_rows_evenly(rows: pd.DataFrame, count: int) -> list[pd.DataFrame]:
    if rows.empty:
        return [rows.copy() for _ in range(count)]
    idx_groups = np.array_split(np.arange(len(rows)), count)
    return [rows.iloc[idx].copy() for idx in idx_groups]


def _fit_tangent(sensor_rows: pd.DataFrame) -> np.ndarray:
    xy = sensor_rows[["x_m", "y_m"]].to_numpy(dtype=float)
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy
    cov = centered.T @ centered / max(len(centered), 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    tangent = eig_vecs[:, int(np.argmax(eig_vals))]

    deltas: list[np.ndarray] = []
    for _run_id, run_group in sensor_rows.groupby("run_id"):
        ordered = run_group.sort_values("timestamp")
        run_xy = ordered[["x_m", "y_m"]].to_numpy(dtype=float)
        if len(run_xy) < 2:
            continue
        step = run_xy[1:] - run_xy[:-1]
        valid = np.linalg.norm(step, axis=1) > 0.5
        if np.any(valid):
            deltas.append(step[valid])
    if deltas:
        mean_delta = np.vstack(deltas).mean(axis=0)
        if float(tangent @ mean_delta) < 0.0:
            tangent = -tangent
    return tangent / np.linalg.norm(tangent)


def build_point_lattice(
    samples_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    config: ContinuityLatticeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ContinuityLatticeConfig()
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])

    rows = samples_df.copy()
    rows["x_m"], rows["y_m"] = latlon_to_xy_m(rows["latitude"], rows["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)

    loop_order = infer_loop_sensor_order(sensor_geometry)
    sensor_meta = sensor_geometry.set_index("node")[["station_id", "cross_m"]]

    point_rows: list[dict[str, object]] = []
    loop_node_index = 0
    for sensor_rank, node in enumerate(loop_order):
        sensor_rows = rows[rows["nearest_sensor"] == node].copy()
        if sensor_rows.empty:
            continue
        tangent = _fit_tangent(sensor_rows)

        center_cut = float(sensor_rows["nearest_sensor_distance_m"].quantile(cfg.center_quantile))
        center_rows = sensor_rows[sensor_rows["nearest_sensor_distance_m"] <= center_cut].copy()
        if center_rows.empty:
            center_rows = sensor_rows.nsmallest(max(1, min(25, len(sensor_rows))), "nearest_sensor_distance_m").copy()
        center_xy = center_rows[["x_m", "y_m"]].to_numpy(dtype=float).mean(axis=0)

        sensor_rows["u_m"] = ((sensor_rows[["x_m", "y_m"]].to_numpy(dtype=float) - center_xy) @ tangent).astype(float)
        neg_rows = sensor_rows[sensor_rows["u_m"] < 0.0].sort_values("u_m").reset_index(drop=True)
        pos_rows = sensor_rows[sensor_rows["u_m"] >= 0.0].sort_values("u_m").reset_index(drop=True)

        half_span = float(np.quantile(np.abs(sensor_rows["u_m"].to_numpy(dtype=float)), 0.85))
        half_span = max(cfg.fallback_half_span_m, min(cfg.max_half_span_m, half_span))

        point_positions: dict[int, np.ndarray] = {3: center_xy}

        for groups, point_ids, sign in [
            (_split_rows_evenly(neg_rows, 2), [1, 2], -1.0),
            (_split_rows_evenly(pos_rows, 2), [4, 5], 1.0),
        ]:
            for group, point_index in zip(groups, point_ids):
                if group.empty:
                    offset = (point_index - 3) * (half_span / 2.0)
                    point_positions[point_index] = center_xy + tangent * offset
                    continue
                point_positions[point_index] = group[["x_m", "y_m"]].to_numpy(dtype=float).mean(axis=0)

        ordered_positions = np.array([point_positions[idx] for idx in range(1, cfg.point_count + 1)], dtype=float)
        projected = ((ordered_positions - center_xy) @ tangent).astype(float)
        order = np.argsort(projected)
        ordered_positions = ordered_positions[order]
        projected = projected[order]

        latitudes, longitudes = xy_to_latlon_m(ordered_positions[:, 0], ordered_positions[:, 1], ref_lat=ref_lat, ref_lon=ref_lon)
        station_id = int(sensor_meta.loc[node, "station_id"])
        side_label = "positive_cross" if float(sensor_meta.loc[node, "cross_m"]) >= 0.0 else "negative_cross"

        for local_idx, (x_m, y_m, lat, lon, u_m) in enumerate(
            zip(ordered_positions[:, 0], ordered_positions[:, 1], latitudes, longitudes, projected),
            start=1,
        ):
            point_rows.append(
                {
                    "sensor_node": node,
                    "station_id": station_id,
                    "side_label": side_label,
                    "sensor_rank": sensor_rank,
                    "point_index": local_idx,
                    "loop_node_index": loop_node_index,
                    "x_m": float(x_m),
                    "y_m": float(y_m),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "u_m": float(u_m),
                    "tangent_x": float(tangent[0]),
                    "tangent_y": float(tangent[1]),
                }
            )
            loop_node_index += 1

    lattice = pd.DataFrame(point_rows).sort_values("loop_node_index").reset_index(drop=True)
    lattice["next_loop_node_index"] = (lattice["loop_node_index"] + 1) % len(lattice)
    lattice["prev_loop_node_index"] = (lattice["loop_node_index"] - 1) % len(lattice)
    return lattice


def build_lattice_edges(lattice_df: pd.DataFrame) -> pd.DataFrame:
    edges: list[dict[str, int]] = []
    n_nodes = len(lattice_df)
    for idx in range(n_nodes):
        edges.append({"src_loop_node_index": idx, "dst_loop_node_index": (idx + 1) % n_nodes})
    return pd.DataFrame(edges)


def assign_ground_truth_loop_nodes(samples_df: pd.DataFrame, lattice_df: pd.DataFrame) -> pd.DataFrame:
    ref_lat = float(lattice_df["latitude"].mean())
    ref_lon = float(lattice_df["longitude"].mean())
    rows = samples_df.copy()
    rows["x_m"], rows["y_m"] = latlon_to_xy_m(rows["latitude"], rows["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)

    sensor_points = {
        node: group.sort_values("point_index")[["loop_node_index", "x_m", "y_m"]].to_numpy(dtype=float)
        for node, group in lattice_df.groupby("sensor_node")
    }

    gt_loop_nodes: list[int] = []
    gt_point_index: list[int] = []
    gt_xy_error_m: list[float] = []

    for row in rows.itertuples(index=False):
        points = sensor_points[str(row.nearest_sensor)]
        deltas = points[:, 1:3] - np.array([[float(row.x_m), float(row.y_m)]], dtype=float)
        dists = np.linalg.norm(deltas, axis=1)
        best_idx = int(np.argmin(dists))
        gt_loop_nodes.append(int(points[best_idx, 0]))
        gt_point_index.append(best_idx + 1)
        gt_xy_error_m.append(float(dists[best_idx]))

    rows["gt_loop_node_index"] = gt_loop_nodes
    rows["gt_point_index"] = gt_point_index
    rows["gt_point_xy_error_m"] = gt_xy_error_m
    return rows


def signed_loop_delta(src_idx: int, dst_idx: int, n_nodes: int) -> int:
    raw = dst_idx - src_idx
    half = n_nodes // 2
    if raw > half:
        raw -= n_nodes
    elif raw < -half:
        raw += n_nodes
    return raw
