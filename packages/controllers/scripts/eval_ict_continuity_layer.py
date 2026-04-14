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
import numpy as np
import pandas as pd

from acies.controller.ict_continuity import (
    ContinuityLatticeConfig,
    assign_ground_truth_loop_nodes,
    build_lattice_edges,
    build_point_lattice,
    infer_loop_sensor_order,
    latlon_to_xy_m,
    sensor_for_station_side,
    signed_loop_delta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a continuity layer on top of ICT hybrid_mic predictions.")
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
        default=Path("docs/design/artifacts/ict_tracker_2026-04-12/continuity_layer"),
    )
    parser.add_argument("--base-mode", default="hybrid_mic")
    parser.add_argument("--point-count", type=int, default=5)
    parser.add_argument("--center-quantile", type=float, default=0.15)
    parser.add_argument("--template-weight", type=float, default=1.8)
    parser.add_argument("--anchor-weight", type=float, default=1.1)
    parser.add_argument("--neighbor-anchor-weight", type=float, default=0.55)
    parser.add_argument("--max-step-nodes", type=int, default=3)
    parser.add_argument("--hop-penalty", type=float, default=0.55)
    parser.add_argument("--direction-bonus", type=float, default=0.30)
    parser.add_argument("--direction-penalty", type=float, default=0.45)
    parser.add_argument("--stay-penalty", type=float, default=0.08)
    parser.add_argument("--reset-penalty", type=float, default=2.25)
    parser.add_argument("--margin-scale", type=float, default=1.0)
    parser.add_argument("--lag-steps", type=int, default=10)
    parser.add_argument("--lock-run-direction", action="store_true")
    parser.add_argument("--loop-selection", choices=["first", "best_smoothed"], default="best_smoothed")
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    parser.add_argument("--plot-runs", type=int, nargs="*", default=[0, 1, 3])
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default="default")
    return parser.parse_args()


def _zscore_array(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0 or math.isnan(std):
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([col for col in df.columns if col.endswith("__mic")])


def load_train_manifest(
    path: Path | None,
    experiment_name: str,
) -> tuple[dict[int, list[int]] | None, dict[int, dict[str, str]]]:
    if path is None:
        return None, {}
    manifest = pd.read_csv(path).copy()
    required = {"experiment_name", "eval_run_id", "eval_label", "train_run_id", "train_label", "scope_name"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Train manifest missing columns: {sorted(missing)}")
    manifest = manifest[manifest["experiment_name"] == experiment_name].copy()
    if manifest.empty:
        raise ValueError(f"No train-manifest rows found for experiment_name={experiment_name!r}")

    manifest["eval_run_id"] = pd.to_numeric(manifest["eval_run_id"], errors="raise").astype(int)
    manifest["train_run_id"] = pd.to_numeric(manifest["train_run_id"], errors="raise").astype(int)

    train_runs_by_eval: dict[int, list[int]] = {}
    meta_by_eval: dict[int, dict[str, str]] = {}
    for eval_run_id, group in manifest.groupby("eval_run_id"):
        train_runs = sorted(set(int(v) for v in group["train_run_id"]))
        if not train_runs:
            raise ValueError(f"No train runs listed for eval run {eval_run_id}")
        train_runs_by_eval[int(eval_run_id)] = train_runs
        meta_by_eval[int(eval_run_id)] = {
            "experiment_name": str(group["experiment_name"].iloc[0]),
            "scope_name": str(group["scope_name"].iloc[0]),
        }
    return train_runs_by_eval, meta_by_eval


def _select_train_group(df: pd.DataFrame, run_id: int, train_runs_by_eval: dict[int, list[int]] | None) -> pd.DataFrame:
    if train_runs_by_eval is None:
        return df[df["run_id"] != run_id].copy()
    if run_id not in train_runs_by_eval:
        raise ValueError(f"No manifest-defined train runs for eval run {run_id}")
    train_group = df[df["run_id"].isin(train_runs_by_eval[run_id])].copy()
    if train_group.empty:
        raise ValueError(f"Manifest-selected train group is empty for eval run {run_id}")
    return train_group


def _loop_sensor_mapping(sensor_geometry: pd.DataFrame) -> tuple[list[str], dict[tuple[int, str], str], dict[str, int]]:
    loop_order = infer_loop_sensor_order(sensor_geometry)
    station_side_to_sensor = {}
    for row in sensor_geometry.itertuples(index=False):
        side = "positive_cross" if float(row.cross_m) >= 0.0 else "negative_cross"
        station_side_to_sensor[(int(row.station_id), side)] = str(row.node)
    sensor_to_rank = {node: idx for idx, node in enumerate(loop_order)}
    return loop_order, station_side_to_sensor, sensor_to_rank


def _global_loop_sign(side_label: str, direction: str) -> int:
    if direction == "ambiguous":
        return 0
    if side_label == "negative_cross":
        return 1 if direction == "toward_S4" else -1
    return 1 if direction == "toward_S1" else -1


def _infer_run_direction_sign(ordered: pd.DataFrame) -> int:
    signs = []
    for row in ordered.itertuples(index=False):
        sign = _global_loop_sign(str(row.pred_side), str(row.pred_direction))
        if sign != 0:
            signs.append(sign)
    if not signs:
        return 0
    total = int(np.sign(np.sum(signs)))
    if total == 0:
        return int(np.sign(np.sum(signs[: max(1, len(signs) // 3)])))
    return total


def _anchor_score_vector(
    lattice_df: pd.DataFrame,
    anchor_sensor_rank: int,
    margin: float,
    margin_scale: float,
    same_weight: float,
    neighbor_weight: float,
) -> np.ndarray:
    n_sensors = int(lattice_df["sensor_rank"].max()) + 1
    margin_trust = max(0.2, min(1.5, float(margin) / max(margin_scale, 1e-6)))
    out = []
    for row in lattice_df.itertuples(index=False):
        dist = abs(int(row.sensor_rank) - anchor_sensor_rank)
        dist = min(dist, n_sensors - dist)
        if dist == 0:
            score = same_weight * margin_trust
        elif dist == 1:
            score = neighbor_weight * (2.0 - margin_trust)
        else:
            score = -float(dist) * margin_trust
        out.append(score)
    return np.array(out, dtype=float)


def _template_scores(sample: np.ndarray, templates: np.ndarray) -> np.ndarray:
    dists = np.nanmean(np.square(templates - sample[None, :]), axis=1)
    dists = np.where(np.isnan(dists), np.nanmax(dists[np.isfinite(dists)]) + 1.0 if np.isfinite(dists).any() else 1.0, dists)
    return -dists


def _transition_score(delta: int, expected_sign: int, hop_penalty: float, direction_bonus: float, direction_penalty: float, stay_penalty: float) -> float:
    if delta == 0:
        return -stay_penalty
    score = -hop_penalty * abs(delta)
    if expected_sign != 0:
        if np.sign(delta) == expected_sign:
            score += direction_bonus * abs(delta)
        else:
            score -= direction_penalty * abs(delta)
    return score


def _find_loop_bounds(axis_m: pd.Series) -> tuple[int, int]:
    smooth = axis_m.rolling(window=5, center=True, min_periods=1).mean().reset_index(drop=True)
    values = smooth.to_list()
    minima = []
    for idx in range(10, len(values) - 10):
        window = values[idx - 10 : idx + 11]
        if values[idx] == min(window) and values[idx] < -85:
            if not minima or idx - minima[-1] > 20:
                minima.append(idx)
    for start, end in zip(minima[:-1], minima[1:]):
        if max(values[start : end + 1]) > 85:
            return start, end
    raise ValueError("Could not find a full loop segment")


def _find_all_loop_bounds(axis_m: pd.Series) -> list[tuple[int, int]]:
    smooth = axis_m.rolling(window=5, center=True, min_periods=1).mean().reset_index(drop=True)
    values = smooth.to_list()
    minima = []
    for idx in range(10, len(values) - 10):
        window = values[idx - 10 : idx + 11]
        if values[idx] == min(window) and values[idx] < -85:
            if not minima or idx - minima[-1] > 20:
                minima.append(idx)
    loops = []
    for start, end in zip(minima[:-1], minima[1:]):
        if max(values[start : end + 1]) > 85:
            loops.append((start, end))
    return loops


def _select_loop_bounds(
    smooth_run: pd.DataFrame,
    mode: str,
) -> tuple[int, int]:
    loops = _find_all_loop_bounds(smooth_run["axis_m"])
    if not loops:
        return _find_loop_bounds(smooth_run["axis_m"])
    if mode == "first":
        return loops[0]
    best = None
    best_key = None
    for start, end in loops:
        loop = smooth_run.iloc[start : end + 1]
        key = (float(loop["pred_xy_error_m"].mean()), -float(((loop["pred_station"] == loop["nearest_station"]) & (loop["pred_side"] == loop["side_label"])).mean()))
        if best_key is None or key < best_key:
            best_key = key
            best = (start, end)
    return best


def _unwrap_loop_progress(node_series: pd.Series, n_nodes: int) -> np.ndarray:
    nodes = node_series.to_numpy(dtype=int)
    out = np.zeros(len(nodes), dtype=float)
    if len(nodes) == 0:
        return out
    out[0] = float(nodes[0])
    for idx in range(1, len(nodes)):
        out[idx] = out[idx - 1] + float(signed_loop_delta(int(nodes[idx - 1]), int(nodes[idx]), n_nodes=n_nodes))
    return out


def _estimate_delay_seconds(gt_progress: np.ndarray, pred_progress: np.ndarray, sample_seconds: float = 1.0, max_lag_steps: int = 15) -> float:
    best_lag = 0
    best_err = float("inf")
    n = len(gt_progress)
    # Remove constant offset before estimating temporal lag.
    base_offset = float(np.median(pred_progress - gt_progress))
    pred_centered = pred_progress - base_offset
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            gt = gt_progress[-lag:]
            pred = pred_centered[: n + lag]
        elif lag > 0:
            gt = gt_progress[: n - lag]
            pred = pred_centered[lag:]
        else:
            gt = gt_progress
            pred = pred_centered
        if len(gt) < 10:
            continue
        err = float(np.mean(np.abs(gt - pred)))
        if err < best_err:
            best_err = err
            best_lag = lag
    return float(best_lag * sample_seconds)


def _estimate_progress_offset(gt_progress: np.ndarray, pred_progress: np.ndarray) -> float:
    return float(np.median(pred_progress - gt_progress))


def _map_sensor_anchor_positions(sensor_geometry: pd.DataFrame) -> dict[tuple[int, str], tuple[float, float]]:
    out = {}
    for station_id, group in sensor_geometry.groupby("station_id"):
        ordered = group.sort_values("cross_m").reset_index(drop=True)
        neg = ordered.iloc[0]
        pos = ordered.iloc[-1]
        out[(int(station_id), "negative_cross")] = (float(neg["latitude"]), float(neg["longitude"]))
        out[(int(station_id), "positive_cross")] = (float(pos["latitude"]), float(pos["longitude"]))
    return out


def _base_xy_metrics(df: pd.DataFrame, sensor_geometry: pd.DataFrame) -> pd.DataFrame:
    anchor_map = _map_sensor_anchor_positions(sensor_geometry)
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])

    base = df.copy()
    pred_lat = []
    pred_lon = []
    for row in base.itertuples(index=False):
        lat, lon = anchor_map[(int(row.pred_station), str(row.pred_side))]
        pred_lat.append(lat)
        pred_lon.append(lon)
    base["pred_latitude"] = pred_lat
    base["pred_longitude"] = pred_lon
    base["pred_x_m"], base["pred_y_m"] = latlon_to_xy_m(base["pred_latitude"], base["pred_longitude"], ref_lat=ref_lat, ref_lon=ref_lon)
    base["gt_x_m"], base["gt_y_m"] = latlon_to_xy_m(base["latitude"], base["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)
    base["pred_xy_error_m"] = np.sqrt(np.square(base["pred_x_m"] - base["gt_x_m"]) + np.square(base["pred_y_m"] - base["gt_y_m"]))
    base["pred_step_m"] = base.groupby("run_id").apply(
        lambda group: np.sqrt(np.square(group["pred_x_m"].diff()) + np.square(group["pred_y_m"].diff()))
    ).reset_index(level=0, drop=True)
    return base


def _build_run_templates(train_group: pd.DataFrame, lattice_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    template_cols = ["gt_loop_node_index"] + feature_cols
    train_templates = train_group[template_cols].groupby("gt_loop_node_index")[feature_cols].mean()
    train_templates = train_templates.reindex(lattice_df["loop_node_index"]).interpolate(limit_direction="both").fillna(0.0)
    return train_templates.to_numpy(dtype=float)


def _build_emission_array(
    ordered: pd.DataFrame,
    lattice_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    feature_cols: list[str],
    template_arr: np.ndarray,
    template_weight: float,
    anchor_weight: float,
    neighbor_anchor_weight: float,
    margin_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    _loop_order, station_side_to_sensor, sensor_to_rank = _loop_sensor_mapping(sensor_geometry)
    emissions = []
    template_margins = []
    for row in ordered.itertuples(index=False):
        sample = np.array([getattr(row, col) for col in feature_cols], dtype=float)
        template_raw = _template_scores(sample, template_arr)
        template_score = _zscore_array(template_raw)
        anchor_sensor = station_side_to_sensor[(int(row.pred_station), str(row.pred_side))]
        anchor_rank = sensor_to_rank[anchor_sensor]
        anchor_raw = _anchor_score_vector(
            lattice_df=lattice_df,
            anchor_sensor_rank=anchor_rank,
            margin=float(row.pred_station_margin),
            margin_scale=margin_scale,
            same_weight=anchor_weight,
            neighbor_weight=neighbor_anchor_weight,
        )
        emissions.append(template_weight * template_score + anchor_raw)
        sorted_template = np.sort(template_raw)[::-1]
        template_margins.append(float(sorted_template[0] - sorted_template[1]))
    return np.vstack(emissions), np.array(template_margins, dtype=float)


def _attach_prediction_geometry(df: pd.DataFrame, lattice_df: pd.DataFrame, tracker_mode: str) -> pd.DataFrame:
    out = df.copy()
    node_lookup = lattice_df.set_index("loop_node_index")
    out["pred_point_index"] = out["pred_loop_node_index"].map(node_lookup["point_index"])
    out["pred_sensor_node"] = out["pred_loop_node_index"].map(node_lookup["sensor_node"])
    out["pred_station"] = out["pred_loop_node_index"].map(node_lookup["station_id"])
    out["pred_side"] = out["pred_loop_node_index"].map(node_lookup["side_label"])
    out["pred_latitude"] = out["pred_loop_node_index"].map(node_lookup["latitude"])
    out["pred_longitude"] = out["pred_loop_node_index"].map(node_lookup["longitude"])
    out["pred_x_m"] = out["pred_loop_node_index"].map(node_lookup["x_m"])
    out["pred_y_m"] = out["pred_loop_node_index"].map(node_lookup["y_m"])
    out["pred_xy_error_m"] = np.sqrt(np.square(out["pred_x_m"] - out["gt_x_m"]) + np.square(out["pred_y_m"] - out["gt_y_m"]))
    out["pred_step_m"] = out.groupby("run_id").apply(
        lambda group: np.sqrt(np.square(group["pred_x_m"].diff()) + np.square(group["pred_y_m"].diff()))
    ).reset_index(level=0, drop=True)
    out["tracker_mode"] = tracker_mode
    return out


def decode_continuity(
    hybrid_df: pd.DataFrame,
    lattice_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    feature_cols: list[str],
    template_weight: float,
    anchor_weight: float,
    neighbor_anchor_weight: float,
    max_step_nodes: int,
    hop_penalty: float,
    direction_bonus: float,
    direction_penalty: float,
    stay_penalty: float,
    reset_penalty: float,
    margin_scale: float,
    train_runs_by_eval: dict[int, list[int]] | None = None,
    meta_by_eval: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    frames = []
    for run_id, test_group in hybrid_df.groupby("run_id"):
        train_group = _select_train_group(hybrid_df, run_id=int(run_id), train_runs_by_eval=train_runs_by_eval)
        template_arr = _build_run_templates(train_group, lattice_df=lattice_df, feature_cols=feature_cols)

        ordered = test_group.sort_values("timestamp").reset_index(drop=True).copy()
        emit_arr, template_margin_arr = _build_emission_array(
            ordered=ordered,
            lattice_df=lattice_df,
            sensor_geometry=sensor_geometry,
            feature_cols=feature_cols,
            template_arr=template_arr,
            template_weight=template_weight,
            anchor_weight=anchor_weight,
            neighbor_anchor_weight=neighbor_anchor_weight,
            margin_scale=margin_scale,
        )
        prev_scores = None
        pred_loop_nodes = []
        pred_direction = []
        pred_confidence = []
        pred_reset = []

        n_nodes = len(lattice_df)
        for t, row in enumerate(ordered.itertuples(index=False)):
            emit = emit_arr[t]
            if prev_scores is None:
                scores = emit
                used_reset = np.ones(n_nodes, dtype=bool)
            else:
                scores = np.full(n_nodes, -1e18, dtype=float)
                used_reset = np.zeros(n_nodes, dtype=bool)
                for dst in range(n_nodes):
                    best_score = emit[dst] - reset_penalty
                    best_reset = True
                    for src in range(n_nodes):
                        delta = signed_loop_delta(src, dst, n_nodes=n_nodes)
                        if abs(delta) > max_step_nodes:
                            continue
                        expected_sign = _global_loop_sign(str(row.pred_side), str(row.pred_direction))
                        candidate = prev_scores[src] + emit[dst] + _transition_score(
                            delta=delta,
                            expected_sign=expected_sign,
                            hop_penalty=hop_penalty,
                            direction_bonus=direction_bonus,
                            direction_penalty=direction_penalty,
                            stay_penalty=stay_penalty,
                        )
                        if candidate > best_score:
                            best_score = candidate
                            best_reset = False
                    scores[dst] = best_score
                    used_reset[dst] = best_reset

            scores = scores - float(np.max(scores))
            best_idx = int(np.argmax(scores))
            pred_loop_nodes.append(int(lattice_df.iloc[best_idx]["loop_node_index"]))
            pred_direction.append(str(row.pred_direction))
            pred_confidence.append(float(scores[best_idx] - np.partition(scores, -2)[-2]))
            pred_reset.append(bool(used_reset[best_idx]))
            prev_scores = scores

        ordered["pred_loop_node_index"] = pred_loop_nodes
        ordered["pred_direction"] = pred_direction
        ordered["pred_station_margin"] = template_margin_arr
        ordered["cc_confidence"] = pred_confidence
        ordered["cc_reset"] = pred_reset
        if meta_by_eval is not None and int(run_id) in meta_by_eval:
            ordered["experiment_name"] = str(meta_by_eval[int(run_id)]["experiment_name"])
            ordered["scope_name"] = str(meta_by_eval[int(run_id)]["scope_name"])
        frames.append(ordered)

    return _attach_prediction_geometry(pd.concat(frames, ignore_index=True), lattice_df=lattice_df, tracker_mode="causal_hybrid_mic")


def decode_smoothed_continuity(
    hybrid_df: pd.DataFrame,
    lattice_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    feature_cols: list[str],
    template_weight: float,
    anchor_weight: float,
    neighbor_anchor_weight: float,
    max_step_nodes: int,
    hop_penalty: float,
    direction_bonus: float,
    direction_penalty: float,
    stay_penalty: float,
    reset_penalty: float,
    margin_scale: float,
    lock_run_direction: bool = False,
    train_runs_by_eval: dict[int, list[int]] | None = None,
    meta_by_eval: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    frames = []
    n_nodes = len(lattice_df)
    for run_id, test_group in hybrid_df.groupby("run_id"):
        train_group = _select_train_group(hybrid_df, run_id=int(run_id), train_runs_by_eval=train_runs_by_eval)
        template_arr = _build_run_templates(train_group, lattice_df=lattice_df, feature_cols=feature_cols)
        ordered = test_group.sort_values("timestamp").reset_index(drop=True).copy()
        run_sign = _infer_run_direction_sign(ordered) if lock_run_direction else 0
        emit_arr, template_margin_arr = _build_emission_array(
            ordered=ordered,
            lattice_df=lattice_df,
            sensor_geometry=sensor_geometry,
            feature_cols=feature_cols,
            template_arr=template_arr,
            template_weight=template_weight,
            anchor_weight=anchor_weight,
            neighbor_anchor_weight=neighbor_anchor_weight,
            margin_scale=margin_scale,
        )

        n_steps = len(ordered)
        dp = np.full((n_steps, n_nodes), -1e18, dtype=float)
        back = np.full((n_steps, n_nodes), -1, dtype=int)
        used_reset = np.zeros((n_steps, n_nodes), dtype=bool)
        dp[0] = emit_arr[0]
        used_reset[0, :] = True

        for t in range(1, n_steps):
            expected_sign = run_sign if run_sign != 0 else _global_loop_sign(str(ordered.iloc[t]["pred_side"]), str(ordered.iloc[t]["pred_direction"]))
            for dst in range(n_nodes):
                best_score = emit_arr[t, dst] - reset_penalty
                best_src = -1
                best_reset = True
                for src in range(n_nodes):
                    delta = signed_loop_delta(src, dst, n_nodes=n_nodes)
                    if abs(delta) > max_step_nodes:
                        continue
                    if run_sign != 0 and delta != 0 and np.sign(delta) != run_sign:
                        continue
                    candidate = dp[t - 1, src] + emit_arr[t, dst] + _transition_score(
                        delta=delta,
                        expected_sign=expected_sign,
                        hop_penalty=hop_penalty,
                        direction_bonus=direction_bonus,
                        direction_penalty=direction_penalty,
                        stay_penalty=stay_penalty,
                    )
                    if candidate > best_score:
                        best_score = candidate
                        best_src = src
                        best_reset = False
                dp[t, dst] = best_score
                back[t, dst] = best_src
                used_reset[t, dst] = best_reset

        state_seq = np.zeros(n_steps, dtype=int)
        state_seq[-1] = int(np.argmax(dp[-1]))
        for t in range(n_steps - 2, -1, -1):
            prev = back[t + 1, state_seq[t + 1]]
            state_seq[t] = prev if prev >= 0 else state_seq[t + 1]

        margins = []
        for t in range(n_steps):
            row_scores = dp[t] - np.max(dp[t])
            margins.append(float(row_scores[state_seq[t]] - np.partition(row_scores, -2)[-2]))

        ordered["pred_loop_node_index"] = state_seq
        ordered["pred_station_margin"] = template_margin_arr
        ordered["pred_direction"] = ordered["pred_direction"].to_numpy()
        ordered["cc_confidence"] = margins
        ordered["cc_reset"] = [bool(used_reset[t, state_seq[t]]) for t in range(n_steps)]
        ordered["cc_run_direction_sign"] = run_sign
        if meta_by_eval is not None and int(run_id) in meta_by_eval:
            ordered["experiment_name"] = str(meta_by_eval[int(run_id)]["experiment_name"])
            ordered["scope_name"] = str(meta_by_eval[int(run_id)]["scope_name"])
        frames.append(ordered)

    return _attach_prediction_geometry(pd.concat(frames, ignore_index=True), lattice_df=lattice_df, tracker_mode="smoothed_hybrid_mic")


def decode_fixed_lag_smoothed_continuity(
    hybrid_df: pd.DataFrame,
    lattice_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    feature_cols: list[str],
    template_weight: float,
    anchor_weight: float,
    neighbor_anchor_weight: float,
    max_step_nodes: int,
    hop_penalty: float,
    direction_bonus: float,
    direction_penalty: float,
    stay_penalty: float,
    reset_penalty: float,
    margin_scale: float,
    lag_steps: int,
    lock_run_direction: bool = False,
    train_runs_by_eval: dict[int, list[int]] | None = None,
    meta_by_eval: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    if lag_steps <= 0:
        return decode_continuity(
            hybrid_df=hybrid_df,
            lattice_df=lattice_df,
            sensor_geometry=sensor_geometry,
            feature_cols=feature_cols,
            template_weight=template_weight,
            anchor_weight=anchor_weight,
            neighbor_anchor_weight=neighbor_anchor_weight,
            max_step_nodes=max_step_nodes,
            hop_penalty=hop_penalty,
            direction_bonus=direction_bonus,
            direction_penalty=direction_penalty,
            stay_penalty=stay_penalty,
            reset_penalty=reset_penalty,
            margin_scale=margin_scale,
        )

    frames = []
    n_nodes = len(lattice_df)
    for run_id, test_group in hybrid_df.groupby("run_id"):
        train_group = _select_train_group(hybrid_df, run_id=int(run_id), train_runs_by_eval=train_runs_by_eval)
        template_arr = _build_run_templates(train_group, lattice_df=lattice_df, feature_cols=feature_cols)
        ordered = test_group.sort_values("timestamp").reset_index(drop=True).copy()
        run_sign = _infer_run_direction_sign(ordered) if lock_run_direction else 0
        emit_arr, template_margin_arr = _build_emission_array(
            ordered=ordered,
            lattice_df=lattice_df,
            sensor_geometry=sensor_geometry,
            feature_cols=feature_cols,
            template_arr=template_arr,
            template_weight=template_weight,
            anchor_weight=anchor_weight,
            neighbor_anchor_weight=neighbor_anchor_weight,
            margin_scale=margin_scale,
        )

        n_steps = len(ordered)
        dp = np.full((n_steps, n_nodes), -1e18, dtype=float)
        back = np.full((n_steps, n_nodes), -1, dtype=int)
        used_reset = np.zeros((n_steps, n_nodes), dtype=bool)
        dp[0] = emit_arr[0]
        used_reset[0, :] = True

        committed = np.full(n_steps, -1, dtype=int)

        for t in range(1, n_steps):
            expected_sign = run_sign if run_sign != 0 else _global_loop_sign(str(ordered.iloc[t]["pred_side"]), str(ordered.iloc[t]["pred_direction"]))
            for dst in range(n_nodes):
                best_score = emit_arr[t, dst] - reset_penalty
                best_src = -1
                best_reset = True
                for src in range(n_nodes):
                    delta = signed_loop_delta(src, dst, n_nodes=n_nodes)
                    if abs(delta) > max_step_nodes:
                        continue
                    if run_sign != 0 and delta != 0 and np.sign(delta) != run_sign:
                        continue
                    candidate = dp[t - 1, src] + emit_arr[t, dst] + _transition_score(
                        delta=delta,
                        expected_sign=expected_sign,
                        hop_penalty=hop_penalty,
                        direction_bonus=direction_bonus,
                        direction_penalty=direction_penalty,
                        stay_penalty=stay_penalty,
                    )
                    if candidate > best_score:
                        best_score = candidate
                        best_src = src
                        best_reset = False
                dp[t, dst] = best_score
                back[t, dst] = best_src
                used_reset[t, dst] = best_reset

            if t >= lag_steps:
                horizon_state = int(np.argmax(dp[t]))
                state = horizon_state
                for back_t in range(t, t - lag_steps, -1):
                    prev = back[back_t, state]
                    if prev < 0:
                        break
                    state = prev
                committed[t - lag_steps] = state

        final_state = int(np.argmax(dp[n_steps - 1]))
        state_seq = np.full(n_steps, -1, dtype=int)
        state_seq[n_steps - 1] = final_state
        for t in range(n_steps - 2, -1, -1):
            prev = back[t + 1, state_seq[t + 1]]
            state_seq[t] = prev if prev >= 0 else state_seq[t + 1]
        for idx in range(n_steps):
            if committed[idx] >= 0:
                state_seq[idx] = committed[idx]

        margins = []
        resets = []
        for t in range(n_steps):
            row_scores = dp[min(t + lag_steps, n_steps - 1)] - np.max(dp[min(t + lag_steps, n_steps - 1)])
            state = state_seq[t]
            margins.append(float(row_scores[state] - np.partition(row_scores, -2)[-2]))
            resets.append(bool(used_reset[min(t + lag_steps, n_steps - 1), state]))

        ordered["pred_loop_node_index"] = state_seq
        ordered["pred_station_margin"] = template_margin_arr
        ordered["pred_direction"] = ordered["pred_direction"].to_numpy()
        ordered["cc_confidence"] = margins
        ordered["cc_reset"] = resets
        ordered["cc_run_direction_sign"] = run_sign
        if meta_by_eval is not None and int(run_id) in meta_by_eval:
            ordered["experiment_name"] = str(meta_by_eval[int(run_id)]["experiment_name"])
            ordered["scope_name"] = str(meta_by_eval[int(run_id)]["scope_name"])
        frames.append(ordered)

    return _attach_prediction_geometry(
        pd.concat(frames, ignore_index=True),
        lattice_df=lattice_df,
        tracker_mode=f"fixedlag{lag_steps}_smoothed_hybrid_mic",
    )


def apply_fixed_lag_smoother(cont_df: pd.DataFrame, lattice_df: pd.DataFrame, lag_steps: int) -> pd.DataFrame:
    if lag_steps <= 0:
        out = cont_df.copy()
        out["tracker_mode"] = "lag0_causal_hybrid_mic"
        return out

    node_lookup = lattice_df.set_index("loop_node_index")
    n_nodes = len(lattice_df)
    frames = []
    for run_id, group in cont_df.groupby("run_id"):
        ordered = group.sort_values("timestamp").reset_index(drop=True).copy()
        node_seq = ordered["pred_loop_node_index"].to_numpy(dtype=int)
        unwrapped = np.zeros(len(node_seq), dtype=float)
        if len(node_seq) > 0:
            unwrapped[0] = float(node_seq[0])
        for idx in range(1, len(node_seq)):
            unwrapped[idx] = unwrapped[idx - 1] + float(signed_loop_delta(int(node_seq[idx - 1]), int(node_seq[idx]), n_nodes=n_nodes))

        smooth = np.zeros(len(node_seq), dtype=float)
        for idx in range(len(node_seq)):
            end = min(len(node_seq), idx + lag_steps + 1)
            smooth[idx] = float(np.median(unwrapped[idx:end]))
        smooth_idx = np.mod(np.rint(smooth).astype(int), n_nodes)

        ordered["pred_loop_node_index"] = smooth_idx
        ordered["pred_point_index"] = ordered["pred_loop_node_index"].map(node_lookup["point_index"])
        ordered["pred_sensor_node"] = ordered["pred_loop_node_index"].map(node_lookup["sensor_node"])
        ordered["pred_station"] = ordered["pred_loop_node_index"].map(node_lookup["station_id"])
        ordered["pred_side"] = ordered["pred_loop_node_index"].map(node_lookup["side_label"])
        ordered["pred_latitude"] = ordered["pred_loop_node_index"].map(node_lookup["latitude"])
        ordered["pred_longitude"] = ordered["pred_loop_node_index"].map(node_lookup["longitude"])
        ordered["pred_x_m"] = ordered["pred_loop_node_index"].map(node_lookup["x_m"])
        ordered["pred_y_m"] = ordered["pred_loop_node_index"].map(node_lookup["y_m"])
        ordered["pred_xy_error_m"] = np.sqrt(np.square(ordered["pred_x_m"] - ordered["gt_x_m"]) + np.square(ordered["pred_y_m"] - ordered["gt_y_m"]))
        ordered["pred_step_m"] = np.sqrt(np.square(ordered["pred_x_m"].diff()) + np.square(ordered["pred_y_m"].diff()))
        ordered["tracker_mode"] = f"lag{lag_steps}_causal_hybrid_mic"
        ordered["cc_reset"] = False
        frames.append(ordered)
    return pd.concat(frames, ignore_index=True)


def summarize(*frames: pd.DataFrame) -> pd.DataFrame:
    compare = pd.concat(list(frames), ignore_index=True, sort=False)
    rows = []
    group_cols = ["tracker_mode"]
    if "experiment_name" in compare.columns:
        group_cols.append("experiment_name")
    if "scope_name" in compare.columns:
        group_cols.append("scope_name")
    for group_key, group in compare.groupby(group_cols):
        if len(group_cols) == 3:
            mode, experiment_name, scope_name = group_key
        elif len(group_cols) == 2:
            mode, experiment_name = group_key
            scope_name = "default"
        else:
            mode = group_key
            experiment_name = "default"
            scope_name = "default"
        non_amb = group[group["direction"] != "ambiguous"]
        reset_rate = float(group["cc_reset"].mean()) if "cc_reset" in group.columns else float("nan")
        rows.append(
            {
                "tracker_mode": mode,
                "experiment_name": str(experiment_name),
                "scope_name": str(scope_name),
                "station_accuracy": float((group["pred_station"] == group["nearest_station"]).mean()),
                "side_accuracy": float((group["pred_side"] == group["side_label"]).mean()),
                "joint_station_side_accuracy": float(
                    ((group["pred_station"] == group["nearest_station"]) & (group["pred_side"] == group["side_label"])).mean()
                ),
                "direction_accuracy_non_ambiguous": float((non_amb["pred_direction"] == non_amb["direction"]).mean())
                if not non_amb.empty
                else float("nan"),
                "point_accuracy": float((group["pred_loop_node_index"] == group["gt_loop_node_index"]).mean()) if "pred_loop_node_index" in group.columns else float("nan"),
                "mean_xy_error_m": float(group["pred_xy_error_m"].mean()),
                "p95_xy_error_m": float(group["pred_xy_error_m"].quantile(0.95)),
                "mean_step_m": float(group["pred_step_m"].dropna().mean()),
                "p95_step_m": float(group["pred_step_m"].dropna().quantile(0.95)),
                "reset_rate": reset_rate,
                "n_samples": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("joint_station_side_accuracy", ascending=False).reset_index(drop=True)


def per_run_summary(*frames: pd.DataFrame) -> pd.DataFrame:
    compare = pd.concat(list(frames), ignore_index=True, sort=False)
    rows = []
    group_cols = ["tracker_mode", "run_id"]
    if "experiment_name" in compare.columns:
        group_cols.append("experiment_name")
    if "scope_name" in compare.columns:
        group_cols.append("scope_name")
    for group_key, group in compare.groupby(group_cols):
        if len(group_cols) == 4:
            mode, run_id, experiment_name, scope_name = group_key
        elif len(group_cols) == 3:
            mode, run_id, experiment_name = group_key
            scope_name = "default"
        else:
            mode, run_id = group_key
            experiment_name = "default"
            scope_name = "default"
        non_amb = group[group["direction"] != "ambiguous"]
        rows.append(
            {
                "tracker_mode": mode,
                "run_id": int(run_id),
                "label": str(group["label"].iloc[0]),
                "experiment_name": str(experiment_name),
                "scope_name": str(scope_name),
                "station_accuracy": float((group["pred_station"] == group["nearest_station"]).mean()),
                "side_accuracy": float((group["pred_side"] == group["side_label"]).mean()),
                "joint_station_side_accuracy": float(
                    ((group["pred_station"] == group["nearest_station"]) & (group["pred_side"] == group["side_label"])).mean()
                ),
                "direction_accuracy_non_ambiguous": float((non_amb["pred_direction"] == non_amb["direction"]).mean())
                if not non_amb.empty
                else float("nan"),
                "mean_xy_error_m": float(group["pred_xy_error_m"].mean()),
                "p95_xy_error_m": float(group["pred_xy_error_m"].quantile(0.95)),
                "mean_step_m": float(group["pred_step_m"].dropna().mean()),
                "reset_rate": float(group["cc_reset"].mean()) if "cc_reset" in group.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["tracker_mode", "run_id"]).reset_index(drop=True)


def plot_lattice(lattice_df: pd.DataFrame, sensor_geometry: pd.DataFrame, out_path: Path) -> None:
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])
    sensor_x, sensor_y = latlon_to_xy_m(sensor_geometry["latitude"], sensor_geometry["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)

    fig, ax = plt.subplots(figsize=(8, 8))
    for node, group in lattice_df.groupby("sensor_node"):
        ax.plot(group["x_m"], group["y_m"], marker="o", linewidth=1.5, label=node)
    ax.scatter(sensor_x, sensor_y, color="black", s=70, label="sensors")
    for row, sx, sy in zip(sensor_geometry.itertuples(index=False), sensor_x, sensor_y):
        ax.text(sx, sy, row.node, fontsize=8, ha="left", va="bottom")
    ax.set_title("ICT 40-node continuity lattice")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["joint_station_side_accuracy", "direction_accuracy_non_ambiguous", "mean_xy_error_m", "p95_step_m"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        colors = ["tab:orange", "tab:green", "tab:red", "tab:blue"][: len(summary_df)]
        ax.bar(summary_df["tracker_mode"], summary_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loop_comparison(
    cont_df: pd.DataFrame,
    smooth_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    lattice_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    run_id: int,
    loop_selection: str,
) -> dict[str, object]:
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])
    cont_run = cont_df[cont_df["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    smooth_run = smooth_df[smooth_df["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    runtime_run = runtime_df[runtime_df["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    lag_run = lag_df[lag_df["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    start_idx, end_idx = _select_loop_bounds(smooth_run, mode=loop_selection)
    cont_loop = cont_run.iloc[start_idx : end_idx + 1].copy()
    smooth_loop = smooth_run.iloc[start_idx : end_idx + 1].copy()
    runtime_loop = runtime_run.iloc[start_idx : end_idx + 1].copy()
    lag_loop = lag_run.iloc[start_idx : end_idx + 1].copy()

    gt_x, gt_y = latlon_to_xy_m(cont_loop["latitude"], cont_loop["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)
    sensor_x, sensor_y = latlon_to_xy_m(sensor_geometry["latitude"], sensor_geometry["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)

    _ = (gt_x, gt_y, sensor_x, sensor_y, lattice_df)

    return {
        "run_id": int(run_id),
        "label": str(cont_loop["label"].iloc[0]),
        "start_elapsed_s": float(cont_loop["elapsed_s"].iloc[0]),
        "end_elapsed_s": float(cont_loop["elapsed_s"].iloc[-1]),
        "continuity_joint_acc": float(
            ((cont_loop["pred_station"] == cont_loop["nearest_station"]) & (cont_loop["pred_side"] == cont_loop["side_label"])).mean()
        ),
        "smoothed_joint_acc": float(
            ((smooth_loop["pred_station"] == smooth_loop["nearest_station"]) & (smooth_loop["pred_side"] == smooth_loop["side_label"])).mean()
        ),
        "runtime_joint_acc": float(
            ((runtime_loop["pred_station"] == runtime_loop["nearest_station"]) & (runtime_loop["pred_side"] == runtime_loop["side_label"])).mean()
        ),
        "lagged_joint_acc": float(
            ((lag_loop["pred_station"] == lag_loop["nearest_station"]) & (lag_loop["pred_side"] == lag_loop["side_label"])).mean()
        ),
        "continuity_mean_xy_error_m": float(cont_loop["pred_xy_error_m"].mean()),
        "smoothed_mean_xy_error_m": float(smooth_loop["pred_xy_error_m"].mean()),
        "runtime_mean_xy_error_m": float(runtime_loop["pred_xy_error_m"].mean()),
        "lagged_mean_xy_error_m": float(lag_loop["pred_xy_error_m"].mean()),
    }


def plot_loop_clean(
    track_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    run_id: int,
    out_path: Path,
    loop_selection: str,
    plot_label: str,
    timing_meaning: str,
    finalize_lag_steps: int | None = None,
) -> None:
    ref_lat = float(sensor_geometry["ref_latitude"].iloc[0])
    ref_lon = float(sensor_geometry["ref_longitude"].iloc[0])
    run_df = track_df[track_df["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    start_idx, end_idx = _select_loop_bounds(run_df, mode=loop_selection)
    loop_df = run_df.iloc[start_idx : end_idx + 1].copy()
    gt_x, gt_y = latlon_to_xy_m(loop_df["latitude"], loop_df["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)
    sensor_x, sensor_y = latlon_to_xy_m(sensor_geometry["latitude"], sensor_geometry["longitude"], ref_lat=ref_lat, ref_lon=ref_lon)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(gt_x, gt_y, color="tab:blue", linewidth=2.2, label="ground truth GPS")
    ax.plot(loop_df["pred_x_m"], loop_df["pred_y_m"], color="tab:green", linewidth=2.0, alpha=0.95, label=plot_label)
    ax.scatter(sensor_x, sensor_y, color="black", s=70, label="sensors")
    ax.scatter(gt_x.iloc[0], gt_y.iloc[0], color="tab:blue", marker="o", s=60, label="GT start")
    ax.scatter(gt_x.iloc[-1], gt_y.iloc[-1], color="tab:blue", marker="s", s=60, label="GT end")
    ax.scatter(loop_df["pred_x_m"].iloc[0], loop_df["pred_y_m"].iloc[0], color="tab:green", marker="o", s=55, label="Pred start")
    ax.scatter(loop_df["pred_x_m"].iloc[-1], loop_df["pred_y_m"].iloc[-1], color="tab:green", marker="s", s=55, label="Pred end")
    sample_idx = loop_df.index[::10]
    for idx in sample_idx:
        pos = int(loop_df.index.get_loc(idx))
        elapsed = int(round(float(loop_df.iloc[pos]["elapsed_s"] - loop_df["elapsed_s"].iloc[0])))
        ts_label = pd.to_datetime(loop_df.iloc[pos]["timestamp"]).strftime("%H:%M:%S")
        pred_label = f"Pred {elapsed}s\nsample {ts_label}"
        if finalize_lag_steps is not None:
            final_pos = min(pos + int(finalize_lag_steps), len(loop_df) - 1)
            final_ts_label = pd.to_datetime(loop_df.iloc[final_pos]["timestamp"]).strftime("%H:%M:%S")
            pred_label = f"Pred {elapsed}s\nsample {ts_label}\nfinal~{final_ts_label}"
        ax.scatter(gt_x.loc[idx], gt_y.loc[idx], color="tab:blue", s=18, alpha=0.6)
        ax.scatter(loop_df.iloc[pos]["pred_x_m"], loop_df.iloc[pos]["pred_y_m"], color="tab:green", s=18, alpha=0.6)
        ax.text(gt_x.loc[idx], gt_y.loc[idx], f"GT {elapsed}s\n{ts_label}", fontsize=7, color="tab:blue", ha="right", va="bottom")
        ax.text(
            loop_df.iloc[pos]["pred_x_m"],
            loop_df.iloc[pos]["pred_y_m"],
            pred_label,
            fontsize=7,
            color="tab:green",
            ha="left",
            va="top",
        )
    for row, sx, sy in zip(sensor_geometry.itertuples(index=False), sensor_x, sensor_y):
        ax.text(sx, sy, row.node, fontsize=8, ha="left", va="bottom")
    ax.set_title(f"run{run_id} one-loop GT vs {plot_label}\n{timing_meaning}")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loop_timing(
    track_df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    run_id: int,
    out_path: Path,
    loop_selection: str,
    plot_label: str,
    timing_meaning: str,
) -> dict[str, float]:
    _ = sensor_geometry
    run_df = track_df[track_df["run_id"] == run_id].sort_values("timestamp").reset_index(drop=True)
    start_idx, end_idx = _select_loop_bounds(run_df, mode=loop_selection)
    loop_df = run_df.iloc[start_idx : end_idx + 1].copy().reset_index(drop=True)
    n_nodes = int(max(loop_df["gt_loop_node_index"].max(), loop_df["pred_loop_node_index"].max()) + 1)
    gt_progress = _unwrap_loop_progress(loop_df["gt_loop_node_index"], n_nodes=n_nodes)
    pred_progress = _unwrap_loop_progress(loop_df["pred_loop_node_index"], n_nodes=n_nodes)
    offset_nodes = _estimate_progress_offset(gt_progress, pred_progress)
    pred_aligned = pred_progress - offset_nodes
    residual_nodes = pred_aligned - gt_progress
    delay_s = _estimate_delay_seconds(gt_progress, pred_progress, sample_seconds=1.0, max_lag_steps=15)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.2]})
    ax = axes[0]
    elapsed = loop_df["elapsed_s"] - loop_df["elapsed_s"].iloc[0]
    ax.plot(elapsed, gt_progress, color="tab:blue", linewidth=2.0, label="GT progress")
    ax.plot(elapsed, pred_aligned, color="tab:green", linewidth=2.0, label="pred progress aligned")
    ax.set_title(f"run{run_id} {plot_label} progress vs time (estimated delay {delay_s:+.0f}s)\n{timing_meaning}")
    ax.set_ylabel("Unwrapped loop node progress")
    ax.legend(loc="best")
    ax_res = axes[1]
    ax_res.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax_res.plot(elapsed, residual_nodes, color="tab:orange", linewidth=1.6, label="aligned progress error")
    ax_res.set_xlabel("Elapsed time within loop (s)")
    ax_res.set_ylabel("Residual\n(nodes)")
    ax_res.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"estimated_delay_s": delay_s}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_runs_by_eval, meta_by_eval = load_train_manifest(args.train_manifest, args.experiment_name)

    pred_df = pd.read_csv(args.predictions)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    sensor_geometry = pd.read_csv(args.sensor_geometry)
    hybrid_df = pred_df[pred_df["tracker_mode"] == args.base_mode].copy()
    if args.runs:
        hybrid_df = hybrid_df[hybrid_df["run_id"].isin(args.runs)].copy()
    hybrid_df = hybrid_df.sort_values(["run_id", "timestamp"]).reset_index(drop=True)

    cfg = ContinuityLatticeConfig(point_count=args.point_count, center_quantile=args.center_quantile)
    lattice_df = build_point_lattice(hybrid_df, sensor_geometry=sensor_geometry, config=cfg)
    lattice_df.to_csv(args.out_dir / "lattice_points.csv", index=False)
    build_lattice_edges(lattice_df).to_csv(args.out_dir / "lattice_edges.csv", index=False)

    labeled_df = assign_ground_truth_loop_nodes(hybrid_df, lattice_df)
    labeled_df["gt_x_m"], labeled_df["gt_y_m"] = latlon_to_xy_m(
        labeled_df["latitude"],
        labeled_df["longitude"],
        ref_lat=float(sensor_geometry["ref_latitude"].iloc[0]),
        ref_lon=float(sensor_geometry["ref_longitude"].iloc[0]),
    )

    feature_cols = _feature_columns(labeled_df)
    base_df = _base_xy_metrics(labeled_df, sensor_geometry=sensor_geometry)
    base_df["pred_loop_node_index"] = -1
    base_df["pred_point_index"] = np.nan
    base_df["pred_sensor_node"] = base_df.apply(
        lambda row: sensor_for_station_side(sensor_geometry, int(row["pred_station"]), str(row["pred_side"])),
        axis=1,
    )
    base_df["cc_reset"] = False
    base_df["cc_confidence"] = np.nan
    base_df["experiment_name"] = args.experiment_name
    base_df["scope_name"] = args.experiment_name

    cont_df = decode_continuity(
        hybrid_df=labeled_df,
        lattice_df=lattice_df,
        sensor_geometry=sensor_geometry,
        feature_cols=feature_cols,
        template_weight=args.template_weight,
        anchor_weight=args.anchor_weight,
        neighbor_anchor_weight=args.neighbor_anchor_weight,
        max_step_nodes=args.max_step_nodes,
        hop_penalty=args.hop_penalty,
        direction_bonus=args.direction_bonus,
        direction_penalty=args.direction_penalty,
        stay_penalty=args.stay_penalty,
        reset_penalty=args.reset_penalty,
        margin_scale=args.margin_scale,
        train_runs_by_eval=train_runs_by_eval,
        meta_by_eval=meta_by_eval,
    )
    smooth_df = decode_smoothed_continuity(
        hybrid_df=labeled_df,
        lattice_df=lattice_df,
        sensor_geometry=sensor_geometry,
        feature_cols=feature_cols,
        template_weight=args.template_weight,
        anchor_weight=args.anchor_weight,
        neighbor_anchor_weight=args.neighbor_anchor_weight,
        max_step_nodes=args.max_step_nodes,
        hop_penalty=args.hop_penalty,
        direction_bonus=args.direction_bonus,
        direction_penalty=args.direction_penalty,
        stay_penalty=args.stay_penalty,
        reset_penalty=args.reset_penalty,
        margin_scale=args.margin_scale,
        train_runs_by_eval=train_runs_by_eval,
        meta_by_eval=meta_by_eval,
    )
    lag_df = apply_fixed_lag_smoother(cont_df, lattice_df=lattice_df, lag_steps=args.lag_steps)
    lag_df["experiment_name"] = args.experiment_name
    lag_df["scope_name"] = args.experiment_name
    runtime_df = decode_fixed_lag_smoothed_continuity(
        hybrid_df=labeled_df,
        lattice_df=lattice_df,
        sensor_geometry=sensor_geometry,
        feature_cols=feature_cols,
        template_weight=args.template_weight,
        anchor_weight=args.anchor_weight,
        neighbor_anchor_weight=args.neighbor_anchor_weight,
        max_step_nodes=args.max_step_nodes,
        hop_penalty=args.hop_penalty,
        direction_bonus=args.direction_bonus,
        direction_penalty=args.direction_penalty,
        stay_penalty=args.stay_penalty,
        reset_penalty=args.reset_penalty,
        margin_scale=args.margin_scale,
        lag_steps=args.lag_steps,
        lock_run_direction=args.lock_run_direction,
        train_runs_by_eval=train_runs_by_eval,
        meta_by_eval=meta_by_eval,
    )

    compare_df = pd.concat([base_df, cont_df, smooth_df, runtime_df, lag_df], ignore_index=True, sort=False)
    compare_df.to_csv(args.out_dir / "continuity_predictions.csv", index=False)

    summary_df = summarize(base_df, cont_df, smooth_df, runtime_df, lag_df)
    per_run_df = per_run_summary(base_df, cont_df, smooth_df, runtime_df, lag_df)
    summary_df.to_csv(args.out_dir / "continuity_summary.csv", index=False)
    per_run_df.to_csv(args.out_dir / "continuity_per_run_summary.csv", index=False)

    plot_lattice(lattice_df, sensor_geometry=sensor_geometry, out_path=args.out_dir / "lattice_overview.png")
    plot_summary(summary_df, out_path=args.out_dir / "continuity_summary.png")

    loop_rows = []
    for run_id in args.plot_runs:
        if run_id not in set(compare_df["run_id"]):
            continue
        loop_rows.append(
            plot_loop_comparison(
                cont_df=cont_df,
                smooth_df=smooth_df,
                runtime_df=runtime_df,
                lag_df=lag_df,
                lattice_df=lattice_df,
                sensor_geometry=sensor_geometry,
                run_id=run_id,
                loop_selection=args.loop_selection,
            )
        )
        plot_loop_clean(
            track_df=smooth_df,
            sensor_geometry=sensor_geometry,
            run_id=run_id,
            out_path=args.out_dir / f"run{run_id}_smoothed_continuity_clean_xy.png",
            loop_selection=args.loop_selection,
            plot_label="smoothed continuity",
            timing_meaning="Offline smoother aligned to sample timestamps; may use future evidence.",
        )
        timing = plot_loop_timing(
            track_df=smooth_df,
            sensor_geometry=sensor_geometry,
            run_id=run_id,
            out_path=args.out_dir / f"run{run_id}_smoothed_continuity_timing.png",
            loop_selection=args.loop_selection,
            plot_label="smoothed continuity",
            timing_meaning="Offline smoother aligned to sample timestamps; delay is not causal wall-clock availability.",
        )
        loop_rows[-1].update(timing)
        loop_rows[-1]["clean_xy_path"] = f"run{run_id}_smoothed_continuity_clean_xy.png"
        loop_rows[-1]["timing_path"] = f"run{run_id}_smoothed_continuity_timing.png"
        plot_loop_clean(
            track_df=runtime_df,
            sensor_geometry=sensor_geometry,
            run_id=run_id,
            out_path=args.out_dir / f"run{run_id}_fixedlag{args.lag_steps}_continuity_clean_xy.png",
            loop_selection=args.loop_selection,
            plot_label=f"fixedlag{args.lag_steps} continuity",
            timing_meaning=f"Bounded-lag runtime-style estimate; sample k is typically finalized when sample k+{args.lag_steps} arrives.",
            finalize_lag_steps=args.lag_steps,
        )
        runtime_timing = plot_loop_timing(
            track_df=runtime_df,
            sensor_geometry=sensor_geometry,
            run_id=run_id,
            out_path=args.out_dir / f"run{run_id}_fixedlag{args.lag_steps}_continuity_timing.png",
            loop_selection=args.loop_selection,
            plot_label=f"fixedlag{args.lag_steps} continuity",
            timing_meaning="Bounded-lag runtime-style estimate; timestamps reflect aligned samples, not zero-lookahead causality.",
        )
        loop_rows[-1]["runtime_clean_xy_path"] = f"run{run_id}_fixedlag{args.lag_steps}_continuity_clean_xy.png"
        loop_rows[-1]["runtime_timing_path"] = f"run{run_id}_fixedlag{args.lag_steps}_continuity_timing.png"
        loop_rows[-1]["runtime_estimated_delay_s"] = runtime_timing["estimated_delay_s"]
        loop_rows[-1]["loop_selection"] = args.loop_selection
    if loop_rows:
        pd.DataFrame(loop_rows).to_csv(args.out_dir / "loop_comparison_summary.csv", index=False)


if __name__ == "__main__":
    main()
