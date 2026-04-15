#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from acies.controller.ict_continuity import assign_ground_truth_loop_nodes, latlon_to_xy_m
from acies.controller.ict_tracker_runtime import (
    DeploymentAssets,
    FeatureNormalizer,
    FixedLagContinuityRuntime,
    StationTemplate,
)

from eval_ict_simple_tracker import (
    apply_corridor_filter,
    build_aligned_dataset,
    build_sensor_geometry,
    compute_feature_file,
    inventory_signal_files,
    load_all_labels,
    load_sensor_locations,
    maybe_load_or_compute,
)

EVAL_SCRIPT_PATH = Path(__file__).resolve().parent / 'eval_ict_continuity_layer.py'
EVAL_SPEC = importlib.util.spec_from_file_location('eval_ict_continuity_layer_script', EVAL_SCRIPT_PATH)
assert EVAL_SPEC is not None and EVAL_SPEC.loader is not None
EVAL_SCRIPT = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(EVAL_SCRIPT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Replay ICT data through the integrated controller continuity runtime.'
    )
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tkimura4/data/2024-03-29-ICT'))
    parser.add_argument('--labels-dir', type=Path, default=Path('docs/design/artifacts/ict_tracker_2026-04-11/labels'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument(
        '--cache-dir', type=Path, default=Path('docs/design/artifacts/ict_tracker_2026-04-14_runtime_integration/cache')
    )
    parser.add_argument('--window-seconds', type=float, default=1.0)
    parser.add_argument('--stride-seconds', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--runs', type=int, nargs='*', default=None)
    parser.add_argument('--plot-runs', type=int, nargs='*', default=[2, 3, 6, 7])
    parser.add_argument('--loop-selection', choices=['first', 'best_smoothed'], default='best_smoothed')
    parser.add_argument('--normalization-mode', choices=['global', 'per_run_oracle'], default='global')
    parser.add_argument('--heldout-by-run', action='store_true')
    parser.add_argument('--max-nearest-sensor-distance-m', type=float, default=None)
    return parser.parse_args()


def _load_aligned_rows(
    data_dir: Path,
    labels_dir: Path,
    cache_dir: Path,
    runs: list[int] | None,
    window_seconds: float,
    stride_seconds: float,
    max_nearest_sensor_distance_m: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_df = load_all_labels(labels_dir, runs=runs)
    items = inventory_signal_files(data_dir)
    if runs is not None:
        run_set = set(runs)
        items = [item for item in items if item.run in run_set]
    features_df = maybe_load_or_compute(
        cache_dir / f'signal_features_w{window_seconds:g}_s{stride_seconds:g}.parquet',
        lambda: pd.concat(
            [
                compute_feature_file(item, window_seconds=window_seconds, stride_seconds=stride_seconds)
                for item in items
            ],
            ignore_index=True,
        ),
    )
    aligned_df = maybe_load_or_compute(
        cache_dir / 'aligned_energy_labels.parquet',
        lambda: build_aligned_dataset(labels_df, features_df, runs=runs),
    )
    aligned_df = apply_corridor_filter(aligned_df, max_nearest_sensor_distance_m)
    return labels_df, aligned_df


def _normalizer_for_run(run_df: pd.DataFrame, feature_names: tuple[str, ...]) -> FeatureNormalizer:
    mean = np.array([float(run_df[name].mean()) for name in feature_names], dtype=np.float64)
    std = np.array([max(float(run_df[name].std(ddof=0)), 1e-9) for name in feature_names], dtype=np.float64)
    return FeatureNormalizer(feature_names=feature_names, mean=mean, std=std)


def _node_feature_stat(
    train_group: pd.DataFrame, lattice_df: pd.DataFrame, feature_names: tuple[str, ...], stat: str
) -> pd.DataFrame:
    template_cols = ['gt_loop_node_index'] + list(feature_names)
    grouped = train_group[template_cols].groupby('gt_loop_node_index')[list(feature_names)]
    if stat == 'mean':
        node_df = grouped.mean()
    elif stat == 'median':
        node_df = grouped.median()
    else:
        raise ValueError(stat)
    return node_df.reindex(lattice_df['loop_node_index']).interpolate(limit_direction='both').fillna(0.0)


def _build_heldout_assets(
    base_assets: DeploymentAssets,
    train_df: pd.DataFrame,
    lattice_df: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> tuple[DeploymentAssets, FeatureNormalizer]:
    normalizer = _normalizer_for_run(train_df, feature_names=feature_names)
    normalized_train = train_df.copy()
    for idx, feature_name in enumerate(feature_names):
        normalized_train[feature_name] = (normalized_train[feature_name] - float(normalizer.mean[idx])) / float(
            normalizer.std[idx]
        )

    station_side_template_df = (
        normalized_train.groupby(['nearest_station', 'side_label'])[list(feature_names)]
        .mean()
        .reset_index()
        .rename(columns={'nearest_station': 'station_id'})
    )
    station_side_templates = tuple(
        StationTemplate(
            station_id=int(row.station_id),
            side_label=str(row.side_label),
            features=np.array([float(getattr(row, name)) for name in feature_names], dtype=np.float64),
        )
        for row in station_side_template_df.itertuples(index=False)
    )
    continuity_templates = _node_feature_stat(
        normalized_train,
        lattice_df=lattice_df,
        feature_names=feature_names,
        stat='median',
    ).to_numpy(dtype=np.float64)

    heldout_assets = DeploymentAssets(
        config=base_assets.config,
        feature_normalizer=FeatureNormalizer.identity(list(feature_names)),
        sensor_order=base_assets.sensor_order,
        sensor_to_station=base_assets.sensor_to_station,
        sensor_to_cross=base_assets.sensor_to_cross,
        station_nodes=base_assets.station_nodes,
        station_side_templates=station_side_templates,
        station_side_states=tuple((item.station_id, item.side_label) for item in station_side_templates),
        station_side_template_matrix=np.vstack([item.features for item in station_side_templates]),
        continuity_templates=continuity_templates,
        lattice_nodes=base_assets.lattice_nodes,
        edges_by_dst=base_assets.edges_by_dst,
        sensor_to_rank=base_assets.sensor_to_rank,
        station_side_to_sensor=base_assets.station_side_to_sensor,
    )
    return heldout_assets, normalizer


def _to_timestamp_ns(value: pd.Timestamp) -> int:
    return int(pd.Timestamp(value).value)


def _summarize_runtime(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tracker_mode, group in pred_df.groupby('tracker_mode'):
        non_amb = group[group['direction'] != 'ambiguous']
        rows.append(
            {
                'tracker_mode': tracker_mode,
                'joint_station_side_accuracy': float(
                    (
                        (group['pred_station'] == group['nearest_station'])
                        & (group['pred_side'] == group['side_label'])
                    ).mean()
                ),
                'direction_accuracy_non_ambiguous': float((non_amb['pred_direction'] == non_amb['direction']).mean())
                if not non_amb.empty
                else float('nan'),
                'point_accuracy': float((group['pred_loop_node_index'] == group['gt_loop_node_index']).mean()),
                'mean_xy_error_m': float(group['pred_xy_error_m'].mean()),
                'p95_xy_error_m': float(group['pred_xy_error_m'].quantile(0.95)),
                'mean_step_m': float(group['pred_step_m'].dropna().mean()),
                'p95_step_m': float(group['pred_step_m'].dropna().quantile(0.95)),
                'reset_rate': float(group['cc_reset'].mean()),
                'n_samples': int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    sensor_geometry = pd.read_csv(args.asset_dir / 'sensor_geometry.csv')
    sensor_geometry.to_csv(args.out_dir / 'sensor_geometry.csv', index=False)

    _labels_df, aligned_df = _load_aligned_rows(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        cache_dir=args.cache_dir,
        runs=args.runs,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        max_nearest_sensor_distance_m=args.max_nearest_sensor_distance_m,
    )

    assets = DeploymentAssets.load(args.asset_dir)
    lattice_df = pd.read_csv(args.asset_dir / 'lattice_points.csv')
    replay_df = assign_ground_truth_loop_nodes(aligned_df.copy(), lattice_df)
    replay_df['gt_x_m'], replay_df['gt_y_m'] = latlon_to_xy_m(
        replay_df['latitude'],
        replay_df['longitude'],
        ref_lat=float(sensor_geometry['ref_latitude'].iloc[0]),
        ref_lon=float(sensor_geometry['ref_longitude'].iloc[0]),
    )

    outputs: list[dict[str, object]] = []
    feature_names = assets.config.feature_names
    mode_suffix = f'{args.normalization_mode}{"_heldout" if args.heldout_by_run else ""}'
    tracker_mode = f'runtime_fixedlag{assets.config.lag_steps}_{mode_suffix}'

    for run_id, run_df in replay_df.groupby('run_id'):
        runtime_assets = assets
        runtime = FixedLagContinuityRuntime(runtime_assets)
        runtime.reset()
        ordered = run_df.sort_values('timestamp').reset_index(drop=True).copy()
        normalizer_override = None
        if args.heldout_by_run:
            train_df = replay_df[replay_df['run_id'] != int(run_id)].copy()
            runtime_assets, normalizer_override = _build_heldout_assets(
                base_assets=assets,
                train_df=train_df,
                lattice_df=lattice_df,
                feature_names=feature_names,
            )
            runtime = FixedLagContinuityRuntime(runtime_assets)
        elif args.normalization_mode == 'per_run_oracle':
            normalizer_override = _normalizer_for_run(ordered, feature_names=feature_names)

        for row in ordered.itertuples(index=False):
            raw_features = np.array([float(getattr(row, name)) for name in feature_names], dtype=np.float64)
            for committed in runtime.step(
                raw_features=raw_features,
                sample_timestamp_ns=_to_timestamp_ns(pd.Timestamp(row.timestamp)),
                finalize_timestamp_ns=_to_timestamp_ns(pd.Timestamp(row.timestamp)),
                label=str(row.label),
                normalizer_override=normalizer_override,
            ):
                outputs.append(
                    {
                        'run_id': int(run_id),
                        'timestamp': pd.Timestamp(committed.sample_timestamp_ns),
                        'finalize_timestamp': pd.Timestamp(committed.finalize_timestamp_ns),
                        'pred_loop_node_index': int(committed.loop_node_index),
                        'pred_station': int(committed.station_id),
                        'pred_side': str(committed.side_label),
                        'pred_direction': str(committed.direction_label),
                        'pred_station_margin': float(committed.station_margin),
                        'cc_confidence': float(committed.confidence),
                        'cc_reset': bool(committed.reset),
                        'pred_latitude': float(committed.latitude),
                        'pred_longitude': float(committed.longitude),
                        'pred_x_m': float(committed.x_m),
                        'pred_y_m': float(committed.y_m),
                    }
                )

        final_ts = _to_timestamp_ns(pd.Timestamp(ordered['timestamp'].iloc[-1]))
        for committed in runtime.flush(finalize_timestamp_ns=final_ts, label=str(ordered['label'].iloc[0])):
            outputs.append(
                {
                    'run_id': int(run_id),
                    'timestamp': pd.Timestamp(committed.sample_timestamp_ns),
                    'finalize_timestamp': pd.Timestamp(committed.finalize_timestamp_ns),
                    'pred_loop_node_index': int(committed.loop_node_index),
                    'pred_station': int(committed.station_id),
                    'pred_side': str(committed.side_label),
                    'pred_direction': str(committed.direction_label),
                    'pred_station_margin': float(committed.station_margin),
                    'cc_confidence': float(committed.confidence),
                    'cc_reset': bool(committed.reset),
                    'pred_latitude': float(committed.latitude),
                    'pred_longitude': float(committed.longitude),
                    'pred_x_m': float(committed.x_m),
                    'pred_y_m': float(committed.y_m),
                }
            )

    pred_only = pd.DataFrame(outputs).sort_values(['run_id', 'timestamp']).reset_index(drop=True)
    compare_df = replay_df.merge(pred_only, on=['run_id', 'timestamp'], how='inner')
    compare_df['tracker_mode'] = tracker_mode
    compare_df['pred_xy_error_m'] = np.sqrt(
        np.square(compare_df['pred_x_m'] - compare_df['gt_x_m'])
        + np.square(compare_df['pred_y_m'] - compare_df['gt_y_m'])
    )
    compare_df['pred_step_m'] = np.sqrt(
        np.square(compare_df.groupby('run_id')['pred_x_m'].diff())
        + np.square(compare_df.groupby('run_id')['pred_y_m'].diff())
    )

    compare_df.to_csv(args.out_dir / 'runtime_predictions.csv', index=False)
    summary_df = _summarize_runtime(compare_df)
    summary_df.to_csv(args.out_dir / 'runtime_summary.csv', index=False)

    per_run_df = EVAL_SCRIPT.per_run_summary(compare_df)
    per_run_df.to_csv(args.out_dir / 'runtime_per_run_summary.csv', index=False)

    loop_rows: list[dict[str, object]] = []
    for run_id in args.plot_runs:
        if run_id not in set(compare_df['run_id']):
            continue
        for pass_idx in range(2):
            mode = f'pass_{pass_idx}'
            clean_xy_path = f'run{run_id}_pass{pass_idx}_controller_runtime_clean_xy.png'
            timing_path = f'run{run_id}_pass{pass_idx}_controller_runtime_timing.png'

            EVAL_SCRIPT.plot_loop_clean(
                track_df=compare_df,
                sensor_geometry=sensor_geometry,
                run_id=int(run_id),
                out_path=args.out_dir / clean_xy_path,
                loop_selection=mode,
                topology=assets.config.topology,
                plot_label=f'controller runtime fixedlag{assets.config.lag_steps}',
                timing_meaning='Integrated controller continuity runtime replayed from ICT 1 s inputs.',
                finalize_lag_steps=assets.config.lag_steps,
            )
            timing = EVAL_SCRIPT.plot_loop_timing(
                track_df=compare_df,
                sensor_geometry=sensor_geometry,
                run_id=int(run_id),
                out_path=args.out_dir / timing_path,
                loop_selection=mode,
                topology=assets.config.topology,
                plot_label=f'controller runtime fixedlag{assets.config.lag_steps}',
                timing_meaning='Integrated controller continuity runtime replayed from ICT 1 s inputs.',
            )
            loop_rows.append(
                {
                    'run_id': int(run_id),
                    'pass_idx': pass_idx,
                    'clean_xy_path': clean_xy_path,
                    'timing_path': timing_path,
                    'estimated_delay_s': timing['estimated_delay_s'],
                    'loop_selection': mode,
                }
            )

    if loop_rows:
        pd.DataFrame(loop_rows).to_csv(args.out_dir / 'runtime_loop_plot_manifest.csv', index=False)


if __name__ == '__main__':
    main()
