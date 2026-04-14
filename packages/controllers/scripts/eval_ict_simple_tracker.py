#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acies.controller.ict_ground_truth import SensorPoint, parse_gps_run_file, project_sensor_positions


@dataclass(frozen=True)
class SensorFile:
    run: int
    node: str
    modality: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a simple offline ICT tracker against discrete ground-truth labels.")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/tkimura4/data/2024-03-29-ICT"))
    parser.add_argument("--labels-dir", type=Path, default=Path("docs/design/artifacts/ict_tracker_2026-04-11/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/design/artifacts/ict_tracker_2026-04-11/simple_tracker"))
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--change-margin-threshold", type=float, default=0.75)
    parser.add_argument("--direction-window", type=int, default=5)
    parser.add_argument("--direction-threshold", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default="default")
    return parser.parse_args()


def coerce_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        out = pd.to_datetime(series, utc=True, errors="coerce")
    elif pd.api.types.is_numeric_dtype(series):
        out = pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    else:
        out = pd.to_datetime(series, utc=True, errors="coerce")
    return out.dt.tz_convert(None)


def inventory_signal_files(data_dir: Path) -> list[SensorFile]:
    rows: list[SensorFile] = []
    for path in sorted(data_dir.glob("run*_rs*_*.parquet")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        run_s, node, modality = parts
        if modality not in {"mic", "geo"}:
            continue
        if not run_s.startswith("run"):
            continue
        rows.append(SensorFile(run=int(run_s[3:]), node=node, modality=modality, path=path))
    return rows


def load_sensor_locations(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / "sensor_location.parquet").copy()
    out = df.rename(columns={"sensor_id": "node", "latitude": "sensor_latitude", "longitude": "sensor_longitude"}).copy()
    out["node"] = out["node"].astype(str)
    return out.sort_values("node").reset_index(drop=True)


def build_sensor_geometry(sensor_locations: pd.DataFrame) -> pd.DataFrame:
    points = [
        SensorPoint(node=row.node, latitude=float(row.sensor_latitude), longitude=float(row.sensor_longitude))
        for row in sensor_locations.itertuples(index=False)
    ]
    return pd.DataFrame(project_sensor_positions(points, pair_gap_threshold_m=10.0))


def infer_time_column(df: pd.DataFrame) -> str:
    for col in ["timestamp", "time", "original_timestamp"]:
        if col in df.columns:
            return col
    raise ValueError(f"Unable to infer time column from {list(df.columns)}")


def infer_signal_column(df: pd.DataFrame) -> str:
    if "samples" in df.columns:
        return "samples"
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(f"Unable to infer signal column from {list(df.columns)}")


def read_signal_parquet(path: Path, modality: str) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    time_col = infer_time_column(df)
    signal_col = infer_signal_column(df)
    out = df[[time_col, signal_col]].rename(columns={time_col: "timestamp", signal_col: "samples"}).copy()
    out["samples"] = pd.to_numeric(out["samples"], errors="coerce")
    out = out.dropna(subset=["timestamp", "samples"]).reset_index(drop=True)

    if modality == "mic" and not out.empty:
        sample_rate = None
        if "sample_rate" in df.columns:
            rate = pd.to_numeric(df["sample_rate"], errors="coerce").dropna()
            if not rate.empty:
                sample_rate = float(rate.iloc[0])
        if sample_rate is None:
            raw_ts = pd.to_numeric(out["timestamp"], errors="coerce")
            deltas = raw_ts.diff().dropna()
            if not deltas.empty and float(deltas.median()) > 0:
                sample_rate = float(1.0 / float(deltas.median()))
        if sample_rate:
            target_rate = 1000.0
            factor = max(1, int(round(sample_rate / target_rate)))
            if factor > 1:
                out = out.iloc[::factor].reset_index(drop=True)

    out["timestamp"] = coerce_datetime(out["timestamp"])
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def build_time_windows(signal_df: pd.DataFrame, window_seconds: float, stride_seconds: float) -> pd.DataFrame:
    if signal_df.empty or len(signal_df) < 2:
        return pd.DataFrame(columns=["window_mid", "log_energy"])

    signal_df = signal_df.sort_values("timestamp").reset_index(drop=True)
    deltas = signal_df["timestamp"].diff().dropna().dt.total_seconds()
    median_dt = float(deltas.median()) if not deltas.empty else np.nan
    if pd.isna(median_dt) or median_dt <= 0:
        return pd.DataFrame(columns=["window_mid", "log_energy"])

    window_size = max(1, int(round(window_seconds / median_dt)))
    stride_size = max(1, int(round(stride_seconds / median_dt)))
    samples = signal_df["samples"].to_numpy(dtype=float)
    timestamps = signal_df["timestamp"].to_numpy(dtype="datetime64[ns]")
    if samples.size < window_size:
        return pd.DataFrame(columns=["window_mid", "log_energy"])

    start_idx = np.arange(0, samples.size - window_size + 1, stride_size)
    end_idx = start_idx + window_size
    square_cumsum = np.concatenate(([0.0], np.cumsum(np.square(samples))))
    mean_square = (square_cumsum[end_idx] - square_cumsum[start_idx]) / window_size
    window_start = pd.to_datetime(timestamps[start_idx])
    window_end = pd.to_datetime(timestamps[end_idx - 1])
    window_mid = window_start + (window_end - window_start) / 2

    return pd.DataFrame(
        {
            "window_mid": window_mid,
            "log_energy": np.log10(mean_square + 1e-12),
        }
    )


def compute_feature_file(item: SensorFile, window_seconds: float, stride_seconds: float) -> pd.DataFrame:
    signal_df = read_signal_parquet(item.path, modality=item.modality)
    feature_df = build_time_windows(signal_df, window_seconds, stride_seconds)
    if feature_df.empty:
        return pd.DataFrame(columns=["run", "node", "modality", "window_mid", "log_energy"])
    feature_df["run"] = item.run
    feature_df["node"] = item.node
    feature_df["modality"] = item.modality
    return feature_df


def maybe_load_or_compute(cache_path: Path, compute_fn):
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    df = compute_fn()
    df.to_parquet(cache_path, index=False)
    return df


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0 or math.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / std


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values * temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    total = np.sum(exp)
    if total == 0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / total


def load_all_labels(labels_dir: Path, runs: list[int] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(labels_dir.glob("run*_discrete_labels.csv")):
        run_id = int(path.stem.split("_")[0][3:])
        if runs is not None and run_id not in runs:
            continue
        df = pd.read_csv(path).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
        frames.append(df)
    if not frames:
        raise ValueError("No label CSVs found")
    return pd.concat(frames, ignore_index=True)


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


def build_aligned_dataset(labels_df: pd.DataFrame, features_df: pd.DataFrame, runs: list[int] | None) -> pd.DataFrame:
    if runs is not None:
        labels_df = labels_df[labels_df["run_id"].isin(runs)].copy()
        features_df = features_df[features_df["run"].isin(runs)].copy()

    feature_frames = []
    for (run_id, node, modality), group in features_df.groupby(["run", "node", "modality"]):
        label_group = labels_df[labels_df["run_id"] == run_id][["timestamp"]].drop_duplicates().sort_values("timestamp")
        if label_group.empty:
            continue
        aligned = pd.merge_asof(
            label_group,
            group.sort_values("window_mid"),
            left_on="timestamp",
            right_on="window_mid",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=0.75),
        )
        aligned["run"] = run_id
        aligned["node"] = node
        aligned["modality"] = modality
        feature_frames.append(aligned[["timestamp", "run", "node", "modality", "log_energy"]])

    long_df = pd.concat(feature_frames, ignore_index=True)
    pivot = (
        long_df.pivot_table(index=["run", "timestamp"], columns=["node", "modality"], values="log_energy", aggfunc="first")
        .sort_index(axis=1)
        .reset_index()
    )
    pivot.columns = ["run", "timestamp"] + [f"{node}__{modality}" for node, modality in pivot.columns.tolist()[2:]]
    return labels_df.merge(pivot, left_on=["run_id", "timestamp"], right_on=["run", "timestamp"], how="inner")


def normalize_energy_columns(df: pd.DataFrame, energy_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in energy_cols:
        out[col] = out.groupby("run_id")[col].transform(zscore)
    return out


def apply_train_normalization(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    for col in feature_cols:
        mean = float(train_df[col].mean())
        std = float(train_df[col].std(ddof=0))
        if std == 0 or math.isnan(std):
            train_out[col] = 0.0
            test_out[col] = 0.0
        else:
            train_out[col] = (train_df[col] - mean) / std
            test_out[col] = (test_df[col] - mean) / std
    return train_out, test_out


def build_tracker_predictions(
    df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    modality_mode: str,
    temperature: float,
    change_margin_threshold: float,
    direction_window: int,
    direction_threshold: float,
) -> pd.DataFrame:
    station_nodes = {
        int(station_id): list(group.sort_values("cross_m")["node"])
        for station_id, group in sensor_geometry.groupby("station_id")
    }

    records = []
    for run_id, group in df.groupby("run_id"):
        group = group.sort_values("timestamp").reset_index(drop=True).copy()
        prev_station = None
        centroid_values = []
        pred_station_list = []
        station_margin_list = []
        side_list = []

        for row in group.itertuples(index=False):
            station_scores = {}
            sensor_scores = {}
            for station_id, nodes in station_nodes.items():
                pair_scores = []
                for node in nodes:
                    mic = getattr(row, f"{node}__mic", np.nan) if f"{node}__mic" in group.columns else np.nan
                    geo = getattr(row, f"{node}__geo", np.nan) if f"{node}__geo" in group.columns else np.nan
                    if modality_mode == "mic":
                        score = mic
                    elif modality_mode == "geo":
                        score = geo
                    else:
                        vals = [x for x in [mic, geo] if not pd.isna(x)]
                        score = float(np.mean(vals)) if vals else np.nan
                    sensor_scores[node] = score
                    if not pd.isna(score):
                        pair_scores.append(score)
                station_scores[station_id] = float(np.nanmax(pair_scores)) if pair_scores else -1e9

            ordered_station_ids = sorted(station_scores)
            ordered_scores = np.array([station_scores[station_id] for station_id in ordered_station_ids], dtype=float)
            weights = softmax(ordered_scores, temperature=temperature)
            station_centroid = float(np.sum(np.array(ordered_station_ids) * weights))
            centroid_values.append(station_centroid)

            ranked = sorted(station_scores.items(), key=lambda item: item[1], reverse=True)
            raw_station = int(ranked[0][0])
            station_margin = float(ranked[0][1] - ranked[1][1])

            if prev_station is None:
                pred_station = raw_station
            elif raw_station == prev_station:
                pred_station = raw_station
            elif abs(raw_station - prev_station) > 1:
                step = 1 if raw_station > prev_station else -1
                pred_station = prev_station + step
            elif station_margin < change_margin_threshold:
                pred_station = prev_station
            else:
                pred_station = raw_station

            pair_nodes = station_nodes[pred_station]
            pair_scored = sorted(
                [(node, sensor_scores[node]) for node in pair_nodes],
                key=lambda item: item[1] if not pd.isna(item[1]) else -1e9,
                reverse=True,
            )
            best_node = pair_scored[0][0]
            best_cross = float(sensor_geometry.set_index("node").loc[best_node, "cross_m"])
            side_label = "positive_cross" if best_cross >= 0 else "negative_cross"

            prev_station = pred_station
            pred_station_list.append(pred_station)
            station_margin_list.append(station_margin)
            side_list.append(side_label)

        group["pred_station"] = pred_station_list
        group["pred_station_margin"] = station_margin_list
        group["pred_side"] = side_list
        group["pred_station_centroid"] = centroid_values
        centroid_delta = group["pred_station_centroid"].diff().rolling(window=direction_window, center=True, min_periods=1).median()
        group["pred_direction"] = "ambiguous"
        group.loc[centroid_delta >= direction_threshold, "pred_direction"] = "toward_S4"
        group.loc[centroid_delta <= -direction_threshold, "pred_direction"] = "toward_S1"
        group["tracker_mode"] = modality_mode
        records.append(group)

    return pd.concat(records, ignore_index=True)


def _mode_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    if mode == "mic":
        return [col for col in df.columns if col.endswith("__mic")]
    if mode == "geo":
        return [col for col in df.columns if col.endswith("__geo")]
    if mode == "fused":
        nodes = sorted({col.split("__")[0] for col in df.columns if "__" in col})
        cols = []
        for node in nodes:
            if f"{node}__mic" in df.columns and f"{node}__geo" in df.columns:
                cols.extend([f"{node}__mic", f"{node}__geo"])
            elif f"{node}__mic" in df.columns:
                cols.append(f"{node}__mic")
            elif f"{node}__geo" in df.columns:
                cols.append(f"{node}__geo")
        return cols
    raise ValueError(mode)


def build_template_tracker_predictions(
    df: pd.DataFrame,
    mode: str,
    direction_window: int,
    direction_threshold: float,
    train_runs_by_eval: dict[int, list[int]] | None = None,
    meta_by_eval: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    feature_cols = _mode_feature_columns(df, mode)
    if not feature_cols:
        raise ValueError(f"No feature columns for mode {mode}")

    frames = []
    for run_id, test_group in df.groupby("run_id"):
        train_group = _select_train_group(df, run_id=int(run_id), train_runs_by_eval=train_runs_by_eval)
        if train_group.empty:
            continue
        templates = (
            train_group.groupby(["nearest_station", "side_label"])[feature_cols]
            .mean()
            .reset_index()
        )
        if templates.empty:
            continue
        template_vectors = templates[feature_cols].to_numpy(dtype=float)
        template_states = list(templates[["nearest_station", "side_label"]].itertuples(index=False, name=None))

        pred_station = []
        pred_side = []
        pred_margin = []
        pred_centroid = []

        ordered = test_group.sort_values("timestamp").reset_index(drop=True).copy()
        for row in ordered.itertuples(index=False):
            sample = np.array([getattr(row, col) for col in feature_cols], dtype=float)
            dists = np.sum(np.square(template_vectors - sample[None, :]), axis=1)
            ranked_idx = np.argsort(dists)
            best_idx = int(ranked_idx[0])
            second_idx = int(ranked_idx[1])
            best_station, best_side = template_states[best_idx]
            pred_station.append(int(best_station))
            pred_side.append(str(best_side))
            pred_margin.append(float(dists[second_idx] - dists[best_idx]))

            station_scores = {}
            for station_id in sorted(set(templates["nearest_station"])):
                station_dists = dists[templates["nearest_station"].to_numpy() == station_id]
                station_scores[int(station_id)] = float(np.min(station_dists))
            inv = np.array([-station_scores[s] for s in sorted(station_scores)], dtype=float)
            weights = softmax(inv, temperature=1.0)
            pred_centroid.append(float(np.sum(np.array(sorted(station_scores)) * weights)))

        ordered["pred_station"] = pred_station
        ordered["pred_side"] = pred_side
        ordered["pred_station_margin"] = pred_margin
        ordered["pred_station_centroid"] = pred_centroid
        centroid_delta = ordered["pred_station_centroid"].diff().rolling(window=direction_window, center=True, min_periods=1).median()
        ordered["pred_direction"] = "ambiguous"
        ordered.loc[centroid_delta >= direction_threshold, "pred_direction"] = "toward_S4"
        ordered.loc[centroid_delta <= -direction_threshold, "pred_direction"] = "toward_S1"
        ordered["tracker_mode"] = f"template_{mode}"
        if meta_by_eval is not None and int(run_id) in meta_by_eval:
            ordered["experiment_name"] = str(meta_by_eval[int(run_id)]["experiment_name"])
            ordered["scope_name"] = str(meta_by_eval[int(run_id)]["scope_name"])
        frames.append(ordered)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["pred_station", "pred_side", "pred_station_margin", "pred_station_centroid", "pred_direction", "tracker_mode"])
    return pd.concat(frames, ignore_index=True)


def _direction_from_station_series(station_series: pd.Series, window: int) -> pd.Series:
    diffs = station_series.diff().fillna(0.0)
    smooth = diffs.rolling(window=window, center=True, min_periods=1).median()
    direction = pd.Series(["ambiguous"] * len(station_series), index=station_series.index)
    direction.loc[smooth > 0] = "toward_S4"
    direction.loc[smooth < 0] = "toward_S1"
    return direction


def build_viterbi_tracker_predictions(
    df: pd.DataFrame,
    mode: str,
    direction_window: int,
    train_runs_by_eval: dict[int, list[int]] | None = None,
    meta_by_eval: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    feature_cols = _mode_feature_columns(df, mode)
    if not feature_cols:
        raise ValueError(f"No feature columns for mode {mode}")

    side_order = ["negative_cross", "positive_cross"]
    states = [(station, side) for station in [1, 2, 3, 4] for side in side_order]
    state_to_idx = {state: idx for idx, state in enumerate(states)}

    frames = []
    for run_id, raw_test_group in df.groupby("run_id"):
        raw_train_group = _select_train_group(df, run_id=int(run_id), train_runs_by_eval=train_runs_by_eval)
        if raw_train_group.empty:
            continue

        train_group, test_group = apply_train_normalization(raw_train_group, raw_test_group.copy(), feature_cols)

        template_df = (
            train_group.groupby(["nearest_station", "side_label"])[feature_cols]
            .mean()
            .reindex(pd.MultiIndex.from_tuples(states, names=["nearest_station", "side_label"]))
            .reset_index()
        )
        template_df[feature_cols] = template_df[feature_cols].fillna(0.0)
        templates = template_df[feature_cols].to_numpy(dtype=float)

        # Learn transition counts from train runs, but restrict to local moves.
        trans_counts = np.ones((len(states), len(states)), dtype=float)
        allowed = np.zeros_like(trans_counts, dtype=bool)
        for i, (station_i, _side_i) in enumerate(states):
            for j, (station_j, _side_j) in enumerate(states):
                if abs(station_i - station_j) <= 1:
                    allowed[i, j] = True
        trans_counts[~allowed] = 0.0

        for _train_run_id, run_group in train_group.groupby("run_id"):
            run_states = list(run_group.sort_values("timestamp")[["nearest_station", "side_label"]].itertuples(index=False, name=None))
            for src, dst in zip(run_states[:-1], run_states[1:]):
                trans_counts[state_to_idx[src], state_to_idx[dst]] += 1.0

        trans_probs = np.zeros_like(trans_counts)
        for i in range(len(states)):
            row_sum = np.sum(trans_counts[i])
            if row_sum == 0:
                trans_probs[i, allowed[i]] = 1.0 / np.sum(allowed[i])
            else:
                trans_probs[i] = trans_counts[i] / row_sum
        log_trans = np.full_like(trans_probs, -1e9, dtype=float)
        mask = trans_probs > 0
        log_trans[mask] = np.log(trans_probs[mask])

        ordered = test_group.sort_values("timestamp").reset_index(drop=True).copy()
        emissions = []
        emission_margin = []
        for row in ordered.itertuples(index=False):
            sample = np.array([getattr(row, col) for col in feature_cols], dtype=float)
            dists = np.mean(np.square(templates - sample[None, :]), axis=1)
            ranked = np.sort(dists)
            emission_margin.append(float(ranked[1] - ranked[0]))
            emissions.append(-dists)
        emission_arr = np.vstack(emissions)

        n_steps = emission_arr.shape[0]
        n_states = emission_arr.shape[1]
        dp = np.full((n_steps, n_states), -1e18, dtype=float)
        back = np.zeros((n_steps, n_states), dtype=int)
        dp[0] = emission_arr[0]
        for t in range(1, n_steps):
            for s in range(n_states):
                prev_scores = dp[t - 1] + log_trans[:, s]
                best_prev = int(np.argmax(prev_scores))
                dp[t, s] = prev_scores[best_prev] + emission_arr[t, s]
                back[t, s] = best_prev

        state_seq = np.zeros(n_steps, dtype=int)
        state_seq[-1] = int(np.argmax(dp[-1]))
        for t in range(n_steps - 2, -1, -1):
            state_seq[t] = back[t + 1, state_seq[t + 1]]

        ordered["pred_station"] = [states[idx][0] for idx in state_seq]
        ordered["pred_side"] = [states[idx][1] for idx in state_seq]
        ordered["pred_station_margin"] = emission_margin
        ordered["pred_station_centroid"] = ordered["pred_station"].astype(float)
        ordered["pred_direction"] = _direction_from_station_series(ordered["pred_station"], window=direction_window)
        ordered["tracker_mode"] = f"viterbi_{mode}"
        if meta_by_eval is not None and int(run_id) in meta_by_eval:
            ordered["experiment_name"] = str(meta_by_eval[int(run_id)]["experiment_name"])
            ordered["scope_name"] = str(meta_by_eval[int(run_id)]["scope_name"])
        frames.append(ordered)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["pred_station", "pred_side", "pred_station_margin", "pred_station_centroid", "pred_direction", "tracker_mode"])
    return pd.concat(frames, ignore_index=True)


def build_sensor_centroid_predictions(
    df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    modality_mode: str,
    temperature: float,
    direction_window: int,
) -> pd.DataFrame:
    sensor_positions = sensor_geometry.set_index("node")[["axis_m", "cross_m"]].to_dict("index")
    station_axis = sensor_geometry.groupby("station_id")["station_axis_m"].first().to_dict()
    sensor_nodes = [row.node for row in sensor_geometry.sort_values("axis_m").itertuples(index=False)]

    frames = []
    for run_id, group in df.groupby("run_id"):
        ordered = group.sort_values("timestamp").reset_index(drop=True).copy()
        pred_station = []
        pred_side = []
        pred_margin = []
        pred_axis = []

        for row in ordered.itertuples(index=False):
            scores = []
            for node in sensor_nodes:
                mic = getattr(row, f"{node}__mic", np.nan) if f"{node}__mic" in ordered.columns else np.nan
                geo = getattr(row, f"{node}__geo", np.nan) if f"{node}__geo" in ordered.columns else np.nan
                if modality_mode == "mic":
                    score = mic
                elif modality_mode == "geo":
                    score = geo
                else:
                    vals = [x for x in [mic, geo] if not pd.isna(x)]
                    score = float(np.mean(vals)) if vals else np.nan
                scores.append(float(score) if not pd.isna(score) else -10.0)

            score_arr = np.array(scores, dtype=float)
            weights = softmax(score_arr, temperature=temperature)
            axis_value = float(np.sum(np.array([sensor_positions[node]["axis_m"] for node in sensor_nodes]) * weights))
            cross_value = float(np.sum(np.array([sensor_positions[node]["cross_m"] for node in sensor_nodes]) * weights))
            pred_axis.append(axis_value)
            pred_side.append("positive_cross" if cross_value >= 0 else "negative_cross")

            station_dists = {station_id: abs(axis_value - axis_pos) for station_id, axis_pos in station_axis.items()}
            ranked = sorted(station_dists.items(), key=lambda item: item[1])
            pred_station.append(int(ranked[0][0]))
            pred_margin.append(float(ranked[1][1] - ranked[0][1]))

        ordered["pred_station"] = pred_station
        ordered["pred_side"] = pred_side
        ordered["pred_station_margin"] = pred_margin
        ordered["pred_station_centroid"] = pred_axis
        centroid_delta = ordered["pred_station_centroid"].diff().rolling(window=direction_window, center=True, min_periods=1).median()
        ordered["pred_direction"] = "ambiguous"
        ordered.loc[centroid_delta > 0, "pred_direction"] = "toward_S4"
        ordered.loc[centroid_delta < 0, "pred_direction"] = "toward_S1"
        ordered["tracker_mode"] = f"sensor_centroid_{modality_mode}"
        frames.append(ordered)

    return pd.concat(frames, ignore_index=True)


def _fit_ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n_features = x.shape[1]
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    reg = np.eye(n_features + 1, dtype=float) * alpha
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    return np.linalg.solve(lhs, rhs)


def build_linear_regression_predictions(
    df: pd.DataFrame,
    sensor_geometry: pd.DataFrame,
    mode: str,
    direction_window: int,
    ridge_alpha: float = 1.0,
    train_runs_by_eval: dict[int, list[int]] | None = None,
    meta_by_eval: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    feature_cols = _mode_feature_columns(df, mode)
    if not feature_cols:
        raise ValueError(f"No feature columns for mode {mode}")

    station_axis = sensor_geometry.groupby("station_id")["station_axis_m"].first().to_dict()
    side_target = df["side_label"].map({"negative_cross": -1.0, "positive_cross": 1.0})

    frames = []
    for run_id, raw_test_group in df.groupby("run_id"):
        raw_train_group = _select_train_group(df, run_id=int(run_id), train_runs_by_eval=train_runs_by_eval)
        if raw_train_group.empty:
            continue
        train_group, test_group = apply_train_normalization(raw_train_group, raw_test_group.copy(), feature_cols)

        x_train = train_group[feature_cols].to_numpy(dtype=float)
        x_test = test_group[feature_cols].to_numpy(dtype=float)
        axis_coef = _fit_ridge_regression(x_train, train_group["axis_m"].to_numpy(dtype=float), alpha=ridge_alpha)
        side_coef = _fit_ridge_regression(x_train, side_target.loc[train_group.index].to_numpy(dtype=float), alpha=ridge_alpha)

        x_test_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1), dtype=float)], axis=1)
        axis_hat = x_test_aug @ axis_coef
        side_hat = x_test_aug @ side_coef

        ordered = test_group.sort_values("timestamp").reset_index(drop=True).copy()
        ordered["pred_station_centroid"] = axis_hat
        ordered["pred_side"] = np.where(side_hat >= 0.0, "positive_cross", "negative_cross")
        ordered["pred_station"] = [
            min(station_axis, key=lambda station_id: abs(axis_value - station_axis[station_id])) for axis_value in axis_hat
        ]
        ranked_gaps = []
        for axis_value in axis_hat:
            dists = sorted(abs(axis_value - station_axis[station_id]) for station_id in sorted(station_axis))
            ranked_gaps.append(float(dists[1] - dists[0]))
        ordered["pred_station_margin"] = ranked_gaps
        centroid_delta = ordered["pred_station_centroid"].diff().rolling(window=direction_window, center=True, min_periods=1).median()
        ordered["pred_direction"] = "ambiguous"
        ordered.loc[centroid_delta > 0, "pred_direction"] = "toward_S4"
        ordered.loc[centroid_delta < 0, "pred_direction"] = "toward_S1"
        ordered["tracker_mode"] = f"linear_{mode}"
        if meta_by_eval is not None and int(run_id) in meta_by_eval:
            ordered["experiment_name"] = str(meta_by_eval[int(run_id)]["experiment_name"])
            ordered["scope_name"] = str(meta_by_eval[int(run_id)]["scope_name"])
        frames.append(ordered)

    if not frames:
        return pd.DataFrame(columns=list(df.columns) + ["pred_station", "pred_side", "pred_station_margin", "pred_station_centroid", "pred_direction", "tracker_mode"])
    return pd.concat(frames, ignore_index=True)


def summarize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["tracker_mode"]
    if "experiment_name" in df.columns:
        group_cols.append("experiment_name")
    if "scope_name" in df.columns:
        group_cols.append("scope_name")
    for group_key, group in df.groupby(group_cols):
        if len(group_cols) == 3:
            mode, experiment_name, scope_name = group_key
        elif len(group_cols) == 2:
            mode, experiment_name = group_key
            scope_name = "default"
        else:
            mode = group_key
            experiment_name = "default"
            scope_name = "default"
        station_acc = float((group["pred_station"] == group["nearest_station"]).mean())
        side_acc = float((group["pred_side"] == group["side_label"]).mean())
        joint_acc = float(((group["pred_station"] == group["nearest_station"]) & (group["pred_side"] == group["side_label"])).mean())
        non_amb = group[group["direction"] != "ambiguous"]
        direction_acc = float((non_amb["pred_direction"] == non_amb["direction"]).mean()) if not non_amb.empty else float("nan")
        rows.append(
            {
                "tracker_mode": mode,
                "experiment_name": str(experiment_name),
                "scope_name": str(scope_name),
                "station_accuracy": station_acc,
                "side_accuracy": side_acc,
                "joint_station_side_accuracy": joint_acc,
                "direction_accuracy_non_ambiguous": direction_acc,
                "n_samples": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("joint_station_side_accuracy", ascending=False).reset_index(drop=True)


def per_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["tracker_mode", "run_id"]
    if "experiment_name" in df.columns:
        group_cols.append("experiment_name")
    if "scope_name" in df.columns:
        group_cols.append("scope_name")
    for group_key, group in df.groupby(group_cols):
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
            }
        )
    return pd.DataFrame(rows).sort_values(["tracker_mode", "run_id"]).reset_index(drop=True)


def plot_run_example(df: pd.DataFrame, out_dir: Path, mode: str, run_id: int) -> None:
    group = df[(df["tracker_mode"] == mode) & (df["run_id"] == run_id)].sort_values("timestamp").copy()
    if group.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(group["elapsed_s"], group["nearest_station"], label="true station", linewidth=2)
    axes[0].plot(group["elapsed_s"], group["pred_station"], label="pred station", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("Station")
    axes[0].set_title(f"run{run_id} {mode} tracker")
    axes[0].legend(loc="best")

    side_map = {"negative_cross": -1, "positive_cross": 1}
    axes[1].plot(group["elapsed_s"], group["side_label"].map(side_map), label="true side", linewidth=2)
    axes[1].plot(group["elapsed_s"], group["pred_side"].map(side_map), label="pred side", linewidth=1.5, alpha=0.8)
    axes[1].set_yticks([-1, 1], labels=["negative", "positive"])
    axes[1].set_ylabel("Side")
    axes[1].legend(loc="best")

    axes[2].plot(group["elapsed_s"], group["pred_station_margin"], label="pred station margin", color="tab:green")
    axes[2].set_ylabel("Margin")
    axes[2].set_xlabel("Elapsed time (s)")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / f"run{run_id}_{mode}_tracker_example.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(summary_df))
    width = 0.2
    ax.bar(x - 1.5 * width, summary_df["station_accuracy"], width=width, label="station")
    ax.bar(x - 0.5 * width, summary_df["side_accuracy"], width=width, label="side")
    ax.bar(x + 0.5 * width, summary_df["joint_station_side_accuracy"], width=width, label="joint")
    ax.bar(x + 1.5 * width, summary_df["direction_accuracy_non_ambiguous"], width=width, label="direction")
    ax.set_xticks(x, summary_df["tracker_mode"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Simple tracker accuracy by modality mode")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tracker_accuracy_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_hybrid_mic_predictions(pred_df: pd.DataFrame, margin_threshold: float = 0.70) -> pd.DataFrame:
    mode_frames = {
        mode: frame.sort_values(["run_id", "timestamp"]).reset_index(drop=True)
        for mode, frame in pred_df.groupby("tracker_mode")
    }
    required = ["mic", "template_mic", "sensor_centroid_mic"]
    if not all(mode in mode_frames for mode in required):
        return pd.DataFrame(columns=pred_df.columns)

    mic = mode_frames["mic"].copy()
    tmpl = mode_frames["template_mic"].copy()
    cent = mode_frames["sensor_centroid_mic"].copy()
    hybrid = mic.copy()
    use_fallback = mic["pred_station_margin"] < margin_threshold
    hybrid.loc[use_fallback, "pred_station"] = tmpl.loc[use_fallback, "pred_station"].to_numpy()
    hybrid.loc[use_fallback, "pred_side"] = tmpl.loc[use_fallback, "pred_side"].to_numpy()
    hybrid["pred_direction"] = cent["pred_direction"].to_numpy()
    hybrid["tracker_mode"] = "hybrid_mic"
    return hybrid


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = set(args.runs) if args.runs else None
    train_runs_by_eval, meta_by_eval = load_train_manifest(args.train_manifest, args.experiment_name)
    labels_df = load_all_labels(args.labels_dir, runs=list(selected_runs) if selected_runs else None)
    sensor_locations = load_sensor_locations(args.data_dir)
    sensor_geometry = build_sensor_geometry(sensor_locations)
    sensor_geometry.to_csv(args.out_dir / "sensor_geometry.csv", index=False)

    items = inventory_signal_files(args.data_dir)
    if selected_runs is not None:
        items = [item for item in items if item.run in selected_runs]

    features_df = maybe_load_or_compute(
        cache_dir / f"signal_features_w{args.window_seconds:g}_s{args.stride_seconds:g}.parquet",
        lambda: pd.concat(
            list(
                ThreadPoolExecutor(max_workers=args.workers).map(
                    partial(compute_feature_file, window_seconds=args.window_seconds, stride_seconds=args.stride_seconds),
                    items,
                )
            ),
            ignore_index=True,
        ),
    )

    aligned_df_raw = maybe_load_or_compute(
        cache_dir / "aligned_energy_labels.parquet",
        lambda: build_aligned_dataset(labels_df, features_df, runs=list(selected_runs) if selected_runs else None),
    )

    energy_cols = [col for col in aligned_df_raw.columns if "__" in col]
    aligned_df = normalize_energy_columns(aligned_df_raw, energy_cols)

    prediction_frames = []
    for mode in ["mic", "geo", "fused"]:
        frame = build_tracker_predictions(
            aligned_df,
            sensor_geometry=sensor_geometry,
            modality_mode=mode,
            temperature=args.temperature,
            change_margin_threshold=args.change_margin_threshold,
            direction_window=args.direction_window,
            direction_threshold=args.direction_threshold,
        )
        frame["experiment_name"] = args.experiment_name
        frame["scope_name"] = args.experiment_name
        prediction_frames.append(frame)

        frame = build_sensor_centroid_predictions(
            aligned_df,
            sensor_geometry=sensor_geometry,
            modality_mode=mode,
            temperature=args.temperature,
            direction_window=args.direction_window,
        )
        frame["experiment_name"] = args.experiment_name
        frame["scope_name"] = args.experiment_name
        prediction_frames.append(frame)

        prediction_frames.append(
            build_template_tracker_predictions(
                aligned_df,
                mode=mode,
                direction_window=args.direction_window,
                direction_threshold=args.direction_threshold,
                train_runs_by_eval=train_runs_by_eval,
                meta_by_eval=meta_by_eval,
            )
        )
        prediction_frames.append(
            build_linear_regression_predictions(
                aligned_df_raw,
                sensor_geometry=sensor_geometry,
                mode=mode,
                direction_window=args.direction_window,
                train_runs_by_eval=train_runs_by_eval,
                meta_by_eval=meta_by_eval,
            )
        )
        prediction_frames.append(
            build_viterbi_tracker_predictions(
                aligned_df_raw,
                mode=mode,
                direction_window=args.direction_window,
                train_runs_by_eval=train_runs_by_eval,
                meta_by_eval=meta_by_eval,
            )
        )
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    hybrid_df = build_hybrid_mic_predictions(pred_df)
    if not hybrid_df.empty:
        hybrid_df["experiment_name"] = args.experiment_name
        hybrid_df["scope_name"] = args.experiment_name
        pred_df = pd.concat([pred_df, hybrid_df], ignore_index=True)
    pred_df.to_csv(args.out_dir / "tracker_predictions.csv", index=False)

    summary_df = summarize_predictions(pred_df)
    per_run_df = per_run_summary(pred_df)
    summary_df.to_csv(args.out_dir / "tracker_summary.csv", index=False)
    per_run_df.to_csv(args.out_dir / "tracker_per_run_summary.csv", index=False)

    plot_summary(summary_df, args.out_dir)
    best_mode = str(summary_df.iloc[0]["tracker_mode"])
    for run_id in sorted(pred_df["run_id"].unique())[:2]:
        plot_run_example(pred_df, args.out_dir, best_mode, int(run_id))


if __name__ == "__main__":
    main()
