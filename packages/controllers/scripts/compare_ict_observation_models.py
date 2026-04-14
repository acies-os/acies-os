#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ICT continuity observation models with loop-blocked CV.")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("docs/design/artifacts/ict_tracker_2026-04-11/simple_tracker/tracker_predictions.csv"),
    )
    parser.add_argument(
        "--sensor-geometry",
        type=Path,
        default=Path("docs/design/artifacts/ict_tracker_2026-04-11/simple_tracker/sensor_geometry.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/design/artifacts/ict_tracker_2026-04-14_observation_model_eval"),
    )
    parser.add_argument("--base-mode", default="hybrid_mic")
    parser.add_argument("--runs", type=int, nargs="*", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


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


def _run_command(cmd: list[str], workdir: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def _build_loop_inventory(pred_df: pd.DataFrame, base_mode: str, n_folds: int) -> pd.DataFrame:
    base_df = pred_df[pred_df["tracker_mode"] == base_mode].copy()
    rows: list[dict[str, object]] = []
    next_loop_id = 0
    for run_id, run_df in base_df.groupby("run_id"):
        ordered = run_df.sort_values("timestamp").reset_index(drop=True)
        bounds = _find_all_loop_bounds(ordered["axis_m"])
        for rank, (start_idx, end_idx) in enumerate(bounds):
            loop_df = ordered.iloc[start_idx : end_idx + 1]
            rows.append(
                {
                    "run_id": int(run_id),
                    "label": str(loop_df["label"].iloc[0]),
                    "loop_id": int(next_loop_id),
                    "loop_rank_in_run": int(rank),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_timestamp": pd.to_datetime(loop_df["timestamp"].iloc[0]).isoformat(),
                    "end_timestamp": pd.to_datetime(loop_df["timestamp"].iloc[-1]).isoformat(),
                    "start_elapsed_s": float(loop_df["elapsed_s"].iloc[0]),
                    "end_elapsed_s": float(loop_df["elapsed_s"].iloc[-1]),
                    "n_points": int(len(loop_df)),
                    "fold_id": int(rank % n_folds),
                }
            )
            next_loop_id += 1
    return pd.DataFrame(rows).sort_values(["run_id", "loop_rank_in_run"]).reset_index(drop=True)


def _summarize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tracker_mode, scope_name), group in pred_df.groupby(["tracker_mode", "scope_name"]):
        non_amb = group[group["direction"] != "ambiguous"]
        rows.append(
            {
                "tracker_mode": tracker_mode,
                "scope_name": scope_name,
                "station_accuracy": float((group["pred_station"] == group["nearest_station"]).mean()),
                "side_accuracy": float((group["pred_side"] == group["side_label"]).mean()),
                "joint_station_side_accuracy": float(
                    ((group["pred_station"] == group["nearest_station"]) & (group["pred_side"] == group["side_label"])).mean()
                ),
                "direction_accuracy_non_ambiguous": float((non_amb["pred_direction"] == non_amb["direction"]).mean())
                if not non_amb.empty
                else float("nan"),
                "point_accuracy": float((group["pred_loop_node_index"] == group["gt_loop_node_index"]).mean())
                if "pred_loop_node_index" in group.columns
                else float("nan"),
                "mean_xy_error_m": float(group["pred_xy_error_m"].mean()),
                "p95_xy_error_m": float(group["pred_xy_error_m"].quantile(0.95)),
                "mean_step_m": float(group["pred_step_m"].dropna().mean()),
                "p95_step_m": float(group["pred_step_m"].dropna().quantile(0.95)),
                "jump_rate_20m": float((group["pred_step_m"].dropna() > 20.0).mean()),
                "reset_rate": float(group["cc_reset"].mean()) if "cc_reset" in group.columns else float("nan"),
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["tracker_mode", "scope_name"]).reset_index(drop=True)


def _per_run_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tracker_mode, scope_name, run_id), group in pred_df.groupby(["tracker_mode", "scope_name", "run_id"]):
        rows.append(
            {
                "tracker_mode": tracker_mode,
                "scope_name": scope_name,
                "run_id": int(run_id),
                "label": str(group["label"].iloc[0]),
                "station_accuracy": float((group["pred_station"] == group["nearest_station"]).mean()),
                "side_accuracy": float((group["pred_side"] == group["side_label"]).mean()),
                "joint_station_side_accuracy": float(
                    ((group["pred_station"] == group["nearest_station"]) & (group["pred_side"] == group["side_label"])).mean()
                ),
                "mean_xy_error_m": float(group["pred_xy_error_m"].mean()),
                "p95_xy_error_m": float(group["pred_xy_error_m"].quantile(0.95)),
                "mean_step_m": float(group["pred_step_m"].dropna().mean()),
                "p95_step_m": float(group["pred_step_m"].dropna().quantile(0.95)),
                "jump_rate_20m": float((group["pred_step_m"].dropna() > 20.0).mean()),
                "reset_rate": float(group["cc_reset"].mean()) if "cc_reset" in group.columns else float("nan"),
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["tracker_mode", "scope_name", "run_id"]).reset_index(drop=True)


def _by_label_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tracker_mode, scope_name, label), group in pred_df.groupby(["tracker_mode", "scope_name", "label"]):
        rows.append(
            {
                "tracker_mode": tracker_mode,
                "scope_name": scope_name,
                "label": str(label),
                "station_accuracy": float((group["pred_station"] == group["nearest_station"]).mean()),
                "side_accuracy": float((group["pred_side"] == group["side_label"]).mean()),
                "joint_station_side_accuracy": float(
                    ((group["pred_station"] == group["nearest_station"]) & (group["pred_side"] == group["side_label"])).mean()
                ),
                "mean_xy_error_m": float(group["pred_xy_error_m"].mean()),
                "p95_xy_error_m": float(group["pred_xy_error_m"].quantile(0.95)),
                "mean_step_m": float(group["pred_step_m"].dropna().mean()),
                "p95_step_m": float(group["pred_step_m"].dropna().quantile(0.95)),
                "jump_rate_20m": float((group["pred_step_m"].dropna() > 20.0).mean()),
                "reset_rate": float(group["cc_reset"].mean()) if "cc_reset" in group.columns else float("nan"),
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["tracker_mode", "scope_name", "label"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(args.predictions)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None)
    if args.runs:
        pred_df = pred_df[pred_df["run_id"].isin(args.runs)].copy()

    loop_inventory = _build_loop_inventory(pred_df, base_mode=args.base_mode, n_folds=args.n_folds)
    loop_manifest_path = args.out_dir / "loop_inventory.csv"
    loop_inventory.to_csv(loop_manifest_path, index=False)

    models = ["pooled_mean", "pooled_median", "diagvar_mean", "kmeans2_min"]
    continuity_script = Path("packages/controllers/scripts/eval_ict_continuity_layer.py")
    run_args = [str(run_id) for run_id in sorted(pred_df["run_id"].unique())]

    fold_jobs: list[tuple[list[str], Path]] = []
    for model in models:
        model_root = args.out_dir / model
        model_root.mkdir(parents=True, exist_ok=True)
        for fold_id in range(args.n_folds):
            fold_dir = model_root / f"fold{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_jobs.append(
                (
                    [
                        sys.executable,
                        str(continuity_script),
                        "--predictions",
                        str(args.predictions),
                        "--sensor-geometry",
                        str(args.sensor_geometry),
                        "--out-dir",
                        str(fold_dir),
                        "--base-mode",
                        args.base_mode,
                        "--template-weight",
                        "1.8",
                        "--anchor-weight",
                        "1.0",
                        "--neighbor-anchor-weight",
                        "0.55",
                        "--max-step-nodes",
                        "2",
                        "--hop-penalty",
                        "0.7",
                        "--direction-bonus",
                        "0.35",
                        "--direction-penalty",
                        "0.25",
                        "--stay-penalty",
                        "0.05",
                        "--reset-penalty",
                        "4.0",
                        "--lag-steps",
                        "5",
                        "--loop-selection",
                        "best_smoothed",
                        "--observation-model",
                        model,
                        "--loop-manifest",
                        str(loop_manifest_path),
                        "--eval-fold",
                        str(fold_id),
                        "--experiment-name",
                        f"fold{fold_id}",
                        "--skip-plots",
                        "--runs",
                        *run_args,
                    ],
                    Path.cwd(),
                )
            )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(executor.map(lambda job: _run_command(job[0], job[1]), fold_jobs))

    fold_summary_frames = []
    oof_prediction_frames = []
    per_loop_frames = []
    for model in models:
        model_root = args.out_dir / model
        for fold_id in range(args.n_folds):
            fold_dir = model_root / f"fold{fold_id}"
            summary_df = pd.read_csv(fold_dir / "continuity_summary.csv").copy()
            summary_df["fold_id"] = fold_id
            summary_df["scope_name"] = model
            fold_summary_frames.append(summary_df)

            per_loop_df = pd.read_csv(fold_dir / "continuity_per_loop_summary.csv").copy()
            per_loop_df["fold_id"] = fold_id
            per_loop_df["scope_name"] = model
            per_loop_frames.append(per_loop_df)

            preds = pd.read_csv(fold_dir / "continuity_predictions.csv").copy()
            preds = preds[preds["eval_split"] == "test"].copy()
            preds["fold_id"] = fold_id
            preds["scope_name"] = model
            oof_prediction_frames.append(preds)

    fullfit_jobs: list[tuple[list[str], Path]] = []
    for model in models:
        model_root = args.out_dir / model
        fullfit_dir = model_root / "fullfit"
        fullfit_dir.mkdir(parents=True, exist_ok=True)
        fullfit_jobs.append(
            (
                [
                    sys.executable,
                    str(continuity_script),
                    "--predictions",
                    str(args.predictions),
                    "--sensor-geometry",
                    str(args.sensor_geometry),
                    "--out-dir",
                    str(fullfit_dir),
                    "--base-mode",
                    args.base_mode,
                    "--template-weight",
                    "1.8",
                    "--anchor-weight",
                    "1.0",
                    "--neighbor-anchor-weight",
                    "0.55",
                    "--max-step-nodes",
                    "2",
                    "--hop-penalty",
                    "0.7",
                    "--direction-bonus",
                    "0.35",
                    "--direction-penalty",
                    "0.25",
                    "--stay-penalty",
                    "0.05",
                    "--reset-penalty",
                    "4.0",
                    "--lag-steps",
                    "5",
                    "--loop-selection",
                    "best_smoothed",
                    "--observation-model",
                    model,
                    "--experiment-name",
                    "fullfit",
                    "--runs",
                    *run_args,
                    "--plot-runs",
                    "2",
                    "3",
                    "6",
                    "7",
                ],
                Path.cwd(),
            )
        )

    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(fullfit_jobs))) as executor:
        list(executor.map(lambda job: _run_command(job[0], job[1]), fullfit_jobs))

    fold_summary = pd.concat(fold_summary_frames, ignore_index=True)
    per_loop = pd.concat(per_loop_frames, ignore_index=True)
    oof_predictions = pd.concat(oof_prediction_frames, ignore_index=True)

    overall_summary = _summarize_predictions(oof_predictions)
    per_run = _per_run_summary(oof_predictions)
    by_label = _by_label_summary(oof_predictions)

    fold_summary.to_csv(args.out_dir / "observation_model_fold_summary.csv", index=False)
    per_loop.to_csv(args.out_dir / "observation_model_per_loop.csv", index=False)
    per_run.to_csv(args.out_dir / "observation_model_per_run.csv", index=False)
    by_label.to_csv(args.out_dir / "observation_model_by_label.csv", index=False)
    overall_summary.to_csv(args.out_dir / "observation_model_overall_summary.csv", index=False)
    oof_predictions.to_csv(args.out_dir / "oof_continuity_predictions.csv", index=False)


if __name__ == "__main__":
    main()
