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
    road_polyline: list[tuple[float, float]] | None = None
    target_point_spacing_m: float = 6.0


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
    for station_id, group in sensor_geometry.sort_values('station_id').groupby('station_id'):
        ordered = group.sort_values('cross_m').reset_index(drop=True)
        negative.append(str(ordered.iloc[0]['node']))
        positive.append(str(ordered.iloc[-1]['node']))
    return negative + list(reversed(positive))


def infer_line_sensor_order(sensor_geometry: pd.DataFrame) -> list[str]:
    ordered = sensor_geometry.sort_values(['station_id', 'cross_m', 'node']).reset_index(drop=True)
    return [str(row.node) for row in ordered.itertuples(index=False)]


def infer_sensor_order(sensor_geometry: pd.DataFrame, topology: str = 'loop') -> list[str]:
    if topology == 'line':
        return infer_line_sensor_order(sensor_geometry)
    return infer_loop_sensor_order(sensor_geometry)


def sensor_for_station_side(sensor_geometry: pd.DataFrame, station_id: int, side_label: str) -> str:
    group = sensor_geometry[sensor_geometry['station_id'] == station_id].sort_values('cross_m').reset_index(drop=True)
    if group.empty:
        raise KeyError(f'No sensor geometry for station {station_id}')
    if side_label == 'negative_cross':
        return str(group.iloc[0]['node'])
    if side_label == 'positive_cross':
        return str(group.iloc[-1]['node'])
    raise KeyError(f'Unknown side label {side_label}')


def _split_rows_evenly(rows: pd.DataFrame, count: int) -> list[pd.DataFrame]:
    if rows.empty:
        return [rows.copy() for _ in range(count)]
    idx_groups = np.array_split(np.arange(len(rows)), count)
    return [rows.iloc[idx].copy() for idx in idx_groups]


def _fit_tangent(sensor_rows: pd.DataFrame) -> np.ndarray:
    xy = sensor_rows[['x_m', 'y_m']].to_numpy(dtype=float)
    mean_xy = xy.mean(axis=0)
    centered = xy - mean_xy
    cov = centered.T @ centered / max(len(centered), 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    tangent = eig_vecs[:, int(np.argmax(eig_vals))]

    deltas: list[np.ndarray] = []
    for _run_id, run_group in sensor_rows.groupby('run_id'):
        ordered = run_group.sort_values('timestamp')
        run_xy = ordered[['x_m', 'y_m']].to_numpy(dtype=float)
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
    topology: str = 'loop',
) -> pd.DataFrame:
    cfg = config or ContinuityLatticeConfig()
    ref_lat = float(sensor_geometry['ref_latitude'].iloc[0])
    ref_lon = float(sensor_geometry['ref_longitude'].iloc[0])

    rows = samples_df.copy()
    rows['x_m'], rows['y_m'] = latlon_to_xy_m(rows['latitude'], rows['longitude'], ref_lat=ref_lat, ref_lon=ref_lon)

    loop_order = infer_sensor_order(sensor_geometry, topology=topology)
    sensor_meta = sensor_geometry.set_index('node')[['station_id', 'cross_m']]

    point_rows: list[dict[str, object]] = []
    loop_node_index = 0

    if getattr(cfg, 'road_polyline', None) is not None:
        road_latlon = np.array(cfg.road_polyline, dtype=float)
        road_x, road_y = latlon_to_xy_m(
            pd.Series(road_latlon[:, 0]), pd.Series(road_latlon[:, 1]), ref_lat=ref_lat, ref_lon=ref_lon
        )
        road_xy = np.column_stack([road_x.to_numpy(), road_y.to_numpy()])

        diffs = np.diff(road_xy, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        road_dists = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_len = road_dists[-1]

        def project_to_polyline(xy: np.ndarray) -> float:
            min_dist = float('inf')
            best_dist_along = 0.0
            for i in range(1, len(road_xy)):
                p1, p2 = road_xy[i - 1], road_xy[i]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                L2 = dx * dx + dy * dy
                t = max(0.0, min(1.0, ((xy[0] - p1[0]) * dx + (xy[1] - p1[1]) * dy) / L2)) if L2 != 0 else 0.0
                px = p1[0] + t * dx
                py = p1[1] + t * dy
                dist_sq = (xy[0] - px) ** 2 + (xy[1] - py) ** 2
                if dist_sq < min_dist:
                    min_dist = dist_sq
                    best_dist_along = road_dists[i - 1] + t * segment_lengths[i - 1]
            return best_dist_along

        def get_polyline_point(d: float) -> tuple[np.ndarray, np.ndarray]:
            d = max(0.0, min(total_len, d))
            for i in range(1, len(road_dists)):
                if d <= road_dists[i] or i == len(road_dists) - 1:
                    d0, d1 = road_dists[i - 1], road_dists[i]
                    t = (d - d0) / (d1 - d0) if d1 > d0 else 0.0
                    pt = road_xy[i - 1] + t * diffs[i - 1]
                    tangent = diffs[i - 1] / segment_lengths[i - 1]
                    return pt, tangent
            return road_xy[-1], diffs[-1] / segment_lengths[-1]

        sensor_dists = []
        for node in loop_order:
            node_row = sensor_geometry[sensor_geometry['node'] == node].iloc[0]
            node_xy = np.array([float(node_row['station_x_m']), float(node_row['station_y_m'])])
            d_along = project_to_polyline(node_xy)
            sensor_dists.append(d_along)

        is_forward = sensor_dists[0] < sensor_dists[-1] if len(sensor_dists) > 1 else True

        boundaries = []
        if is_forward:
            boundaries.append(0.0)
            for i in range(1, len(sensor_dists)):
                boundaries.append((sensor_dists[i - 1] + sensor_dists[i]) / 2.0)
            boundaries.append(total_len)
        else:
            boundaries.append(total_len)
            for i in range(1, len(sensor_dists)):
                boundaries.append((sensor_dists[i - 1] + sensor_dists[i]) / 2.0)
            boundaries.append(0.0)

        for sensor_rank, node in enumerate(loop_order):
            b_start = boundaries[sensor_rank]
            b_end = boundaries[sensor_rank + 1]

            length = abs(b_end - b_start)
            num_points = max(1, round(length / cfg.target_point_spacing_m))

            station_id = int(sensor_meta.loc[node, 'station_id'])
            side_label = 'positive_cross' if float(sensor_meta.loc[node, 'cross_m']) >= 0.0 else 'negative_cross'

            step = (b_end - b_start) / num_points

            for i in range(num_points):
                d_pt = b_start + step * (i + 0.5)
                pt, tangent = get_polyline_point(d_pt)

                if b_end < b_start:
                    tangent = -tangent

                lat_arr, lon_arr = xy_to_latlon_m(
                    np.array([pt[0]]), np.array([pt[1]]), ref_lat=ref_lat, ref_lon=ref_lon
                )

                point_rows.append(
                    {
                        'sensor_node': node,
                        'station_id': station_id,
                        'side_label': side_label,
                        'sensor_rank': sensor_rank,
                        'point_index': i + 1,
                        'loop_node_index': loop_node_index,
                        'x_m': float(pt[0]),
                        'y_m': float(pt[1]),
                        'latitude': float(lat_arr[0]),
                        'longitude': float(lon_arr[0]),
                        'u_m': float(d_pt),
                        'tangent_x': float(tangent[0]),
                        'tangent_y': float(tangent[1]),
                    }
                )
                loop_node_index += 1

    else:
        for sensor_rank, node in enumerate(loop_order):
            sensor_rows = rows[rows['nearest_sensor'] == node].copy()
            if sensor_rows.empty:
                continue
            tangent = _fit_tangent(sensor_rows)

            center_cut = float(sensor_rows['nearest_sensor_distance_m'].quantile(cfg.center_quantile))
            center_rows = sensor_rows[sensor_rows['nearest_sensor_distance_m'] <= center_cut].copy()
            if center_rows.empty:
                center_rows = sensor_rows.nsmallest(
                    max(1, min(25, len(sensor_rows))), 'nearest_sensor_distance_m'
                ).copy()
            center_xy = center_rows[['x_m', 'y_m']].to_numpy(dtype=float).mean(axis=0)

            sensor_rows['u_m'] = ((sensor_rows[['x_m', 'y_m']].to_numpy(dtype=float) - center_xy) @ tangent).astype(
                float
            )
            neg_rows = sensor_rows[sensor_rows['u_m'] < 0.0].sort_values('u_m').reset_index(drop=True)
            pos_rows = sensor_rows[sensor_rows['u_m'] >= 0.0].sort_values('u_m').reset_index(drop=True)

            half_span = float(np.quantile(np.abs(sensor_rows['u_m'].to_numpy(dtype=float)), 0.85))
            half_span = max(cfg.fallback_half_span_m, min(cfg.max_half_span_m, half_span))

            if topology == 'line':
                sensor_row = sensor_geometry[sensor_geometry['node'] == node].iloc[0]
                center_xy = np.array([float(sensor_row['station_x_m']), float(sensor_row['station_y_m'])])

            point_positions: dict[int, np.ndarray] = {3: center_xy}

            for groups, point_ids, sign in [
                (_split_rows_evenly(neg_rows, 2), [1, 2], -1.0),
                (_split_rows_evenly(pos_rows, 2), [4, 5], 1.0),
            ]:
                for group, point_index in zip(groups, point_ids):
                    if group.empty or topology == 'line':
                        offset = (point_index - 3) * (half_span / 2.0)
                        point_positions[point_index] = center_xy + tangent * offset
                        continue
                    point_positions[point_index] = group[['x_m', 'y_m']].to_numpy(dtype=float).mean(axis=0)

            ordered_positions = np.array([point_positions[idx] for idx in range(1, cfg.point_count + 1)], dtype=float)
            projected = ((ordered_positions - center_xy) @ tangent).astype(float)
            order = np.argsort(projected)
            ordered_positions = ordered_positions[order]
            projected = projected[order]

            latitudes, longitudes = xy_to_latlon_m(
                ordered_positions[:, 0], ordered_positions[:, 1], ref_lat=ref_lat, ref_lon=ref_lon
            )
            station_id = int(sensor_meta.loc[node, 'station_id'])
            side_label = 'positive_cross' if float(sensor_meta.loc[node, 'cross_m']) >= 0.0 else 'negative_cross'

            for local_idx, (x_m, y_m, lat, lon, u_m) in enumerate(
                zip(ordered_positions[:, 0], ordered_positions[:, 1], latitudes, longitudes, projected),
                start=1,
            ):
                point_rows.append(
                    {
                        'sensor_node': node,
                        'station_id': station_id,
                        'side_label': side_label,
                        'sensor_rank': sensor_rank,
                        'point_index': local_idx,
                        'loop_node_index': loop_node_index,
                        'x_m': float(x_m),
                        'y_m': float(y_m),
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'u_m': float(u_m),
                        'tangent_x': float(tangent[0]),
                        'tangent_y': float(tangent[1]),
                    }
                )
                loop_node_index += 1

    lattice = pd.DataFrame(point_rows).sort_values('loop_node_index').reset_index(drop=True)
    n_nodes = len(lattice)
    if topology == 'line':
        lattice['next_loop_node_index'] = np.minimum(lattice['loop_node_index'] + 1, max(n_nodes - 1, 0))
        lattice['prev_loop_node_index'] = np.maximum(lattice['loop_node_index'] - 1, 0)
    else:
        lattice['next_loop_node_index'] = (lattice['loop_node_index'] + 1) % n_nodes
        lattice['prev_loop_node_index'] = (lattice['loop_node_index'] - 1) % n_nodes
    return lattice


def build_lattice_edges(lattice_df: pd.DataFrame, topology: str = 'loop') -> pd.DataFrame:
    edges: list[dict[str, int]] = []
    n_nodes = len(lattice_df)
    if topology == 'line':
        for idx in range(max(0, n_nodes - 1)):
            edges.append({'src_loop_node_index': idx, 'dst_loop_node_index': idx + 1})
    else:
        for idx in range(n_nodes):
            edges.append({'src_loop_node_index': idx, 'dst_loop_node_index': (idx + 1) % n_nodes})
    return pd.DataFrame(edges)


def assign_ground_truth_loop_nodes(samples_df: pd.DataFrame, lattice_df: pd.DataFrame) -> pd.DataFrame:
    ref_lat = float(lattice_df['latitude'].mean())
    ref_lon = float(lattice_df['longitude'].mean())
    rows = samples_df.copy()
    rows['x_m'], rows['y_m'] = latlon_to_xy_m(rows['latitude'], rows['longitude'], ref_lat=ref_lat, ref_lon=ref_lon)

    sensor_points = {
        node: group.sort_values('point_index')[['loop_node_index', 'x_m', 'y_m']].to_numpy(dtype=float)
        for node, group in lattice_df.groupby('sensor_node')
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

    rows['gt_loop_node_index'] = gt_loop_nodes
    rows['gt_point_index'] = gt_point_index
    rows['gt_point_xy_error_m'] = gt_xy_error_m
    return rows


def signed_loop_delta(src_idx: int, dst_idx: int, n_nodes: int) -> int:
    raw = dst_idx - src_idx
    half = n_nodes // 2
    if raw > half:
        raw -= n_nodes
    elif raw < -half:
        raw += n_nodes
    return raw


def signed_topology_delta(src_idx: int, dst_idx: int, n_nodes: int, topology: str = 'loop') -> int:
    if topology == 'line':
        return dst_idx - src_idx
    return signed_loop_delta(src_idx, dst_idx, n_nodes)
