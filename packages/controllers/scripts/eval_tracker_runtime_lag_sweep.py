#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1] / 'src'
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

EVAL_SCRIPT_PATH = Path(__file__).resolve().parent / 'eval_ict_continuity_layer.py'
EVAL_SPEC = importlib.util.spec_from_file_location('eval_ict_continuity_layer_script', EVAL_SCRIPT_PATH)
assert EVAL_SPEC is not None and EVAL_SPEC.loader is not None
EVAL_SCRIPT = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(EVAL_SCRIPT)

from acies.controller.ict_continuity import assign_ground_truth_loop_nodes, latlon_to_xy_m
from acies.controller.ict_tracker_runtime import (
    DeploymentAssets,
    FixedLagContinuityRuntime,
)

# Import helpers from simple tracker eval
from eval_ict_simple_tracker import (
    build_sensor_geometry,
    load_sensor_locations,
)

# Import loading helpers from the existing replay script
from eval_tracker_runtime_replay import (
    _load_aligned_rows,
    _to_timestamp_ns,
    _summarize_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sweep lag_steps for the integrated controller continuity runtime.')
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tkimura4/data/2024-03-29-ICT'))
    parser.add_argument('--labels-dir', type=Path, default=Path('docs/design/artifacts/ict_tracker_2026-04-11/labels'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument(
        '--cache-dir', type=Path, default=Path('docs/design/artifacts/ict_tracker_2026-04-14_runtime_integration/cache')
    )
    parser.add_argument('--lag-steps-list', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 7, 10])
    parser.add_argument('--window-seconds', type=float, default=1.0)
    parser.add_argument('--stride-seconds', type=float, default=1.0)
    parser.add_argument('--runs', type=int, nargs='*', default=None)
    parser.add_argument('--plot-runs', type=int, nargs='*', default=[2, 6])
    parser.add_argument('--plot-lags', type=int, nargs='*', default=[1, 5, 10])
    parser.add_argument('--loop-selection', choices=['first', 'best_smoothed'], default='best_smoothed')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'Will sweep lag steps: {args.lag_steps_list}')

    sensor_locations = load_sensor_locations(args.data_dir)
    sensor_geometry = build_sensor_geometry(sensor_locations)

    _labels_df, aligned_df = _load_aligned_rows(
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        cache_dir=args.cache_dir,
        runs=args.runs,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
    )

    base_assets = DeploymentAssets.load(args.asset_dir)
    lattice_df = pd.read_csv(args.asset_dir / 'lattice_points.csv')
    replay_df = assign_ground_truth_loop_nodes(aligned_df.copy(), lattice_df)
    replay_df['gt_x_m'], replay_df['gt_y_m'] = latlon_to_xy_m(
        replay_df['latitude'],
        replay_df['longitude'],
        ref_lat=float(sensor_geometry['ref_latitude'].iloc[0]),
        ref_lon=float(sensor_geometry['ref_longitude'].iloc[0]),
    )

    feature_names = base_assets.config.feature_names
    all_summaries = []

    for lag in args.lag_steps_list:
        print(f'Evaluating lag_steps = {lag}...')

        # Create overridden assets for this lag
        new_config = replace(base_assets.config, lag_steps=lag)
        sweep_assets = replace(base_assets, config=new_config)

        outputs = []
        for run_id, run_df in replay_df.groupby('run_id'):
            runtime = FixedLagContinuityRuntime(sweep_assets)
            ordered = run_df.sort_values('timestamp').reset_index(drop=True)

            for row in ordered.itertuples(index=False):
                raw_features = np.array([float(getattr(row, name)) for name in feature_names], dtype=np.float64)
                for committed in runtime.step(
                    raw_features=raw_features,
                    sample_timestamp_ns=_to_timestamp_ns(pd.Timestamp(row.timestamp)),
                    finalize_timestamp_ns=_to_timestamp_ns(pd.Timestamp(row.timestamp)),
                    label=str(row.label),
                ):
                    outputs.append(
                        {
                            'run_id': int(run_id),
                            'timestamp': pd.Timestamp(committed.sample_timestamp_ns),
                            'pred_loop_node_index': int(committed.loop_node_index),
                            'pred_station': int(committed.station_id),
                            'pred_side': str(committed.side_label),
                            'pred_direction': str(committed.direction_label),
                            'pred_x_m': float(committed.x_m),
                            'pred_y_m': float(committed.y_m),
                            'cc_reset': bool(committed.reset),
                        }
                    )

            final_ts = _to_timestamp_ns(pd.Timestamp(ordered['timestamp'].iloc[-1]))
            for committed in runtime.flush(finalize_timestamp_ns=final_ts, label=str(ordered['label'].iloc[0])):
                outputs.append(
                    {
                        'run_id': int(run_id),
                        'timestamp': pd.Timestamp(committed.sample_timestamp_ns),
                        'pred_loop_node_index': int(committed.loop_node_index),
                        'pred_station': int(committed.station_id),
                        'pred_side': str(committed.side_label),
                        'pred_direction': str(committed.direction_label),
                        'pred_x_m': float(committed.x_m),
                        'pred_y_m': float(committed.y_m),
                        'cc_reset': bool(committed.reset),
                    }
                )

        pred_df = pd.DataFrame(outputs)
        compare_df = replay_df.merge(pred_df, on=['run_id', 'timestamp'], how='inner')
        compare_df['tracker_mode'] = f'lag_{lag}'
        compare_df['pred_xy_error_m'] = np.sqrt(
            np.square(compare_df['pred_x_m'] - compare_df['gt_x_m'])
            + np.square(compare_df['pred_y_m'] - compare_df['gt_y_m'])
        )
        compare_df['pred_step_m'] = (
            compare_df.groupby('run_id')
            .apply(lambda group: np.sqrt(np.square(group['pred_x_m'].diff()) + np.square(group['pred_y_m'].diff())))
            .reset_index(level=0, drop=True)
        )

        for run_id in args.plot_runs:
            if lag in args.plot_lags and run_id in set(compare_df['run_id']):
                out_path = args.out_dir / f'run{run_id}_lag{lag}_clean_xy.png'
                EVAL_SCRIPT.plot_loop_clean(
                    track_df=compare_df,
                    sensor_geometry=sensor_geometry,
                    run_id=int(run_id),
                    out_path=out_path,
                    loop_selection=args.loop_selection,
                    plot_label=f'lag_steps={lag}',
                    timing_meaning='Lag sweep visualization',
                    finalize_lag_steps=lag,
                )
                print(f'Saved plot {out_path}')

        summary = _summarize_runtime(compare_df)
        summary['lag_steps'] = lag
        all_summaries.append(summary)

    final_summary_df = pd.concat(all_summaries, ignore_index=True)
    final_summary_df.to_csv(args.out_dir / 'lag_sweep_summary.csv', index=False)
    print(f'Saved summary to {args.out_dir / "lag_sweep_summary.csv"}')

    # Plot Accuracy and Error vs Lag Steps
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_acc = 'tab:blue'
    ax1.set_xlabel('Lag Steps (Seconds Delay)')
    ax1.set_ylabel('Joint Station+Side Accuracy', color=color_acc)
    ax1.plot(
        final_summary_df['lag_steps'],
        final_summary_df['joint_station_side_accuracy'],
        marker='o',
        color=color_acc,
        linewidth=2,
    )
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_xticks(args.lag_steps_list)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color_err = 'tab:red'
    ax2.set_ylabel('Mean XY Error (m)', color=color_err)
    ax2.plot(
        final_summary_df['lag_steps'],
        final_summary_df['mean_xy_error_m'],
        marker='s',
        color=color_err,
        linewidth=2,
        linestyle='--',
    )
    ax2.tick_params(axis='y', labelcolor=color_err)

    plt.title('Tracker Performance vs. Fixed-Lag Delay')
    fig.tight_layout()

    plot_path = args.out_dir / 'lag_vs_accuracy.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Saved tradeoff plot to {plot_path}')


if __name__ == '__main__':
    main()
