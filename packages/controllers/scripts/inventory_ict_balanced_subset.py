#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inventory ICT runs and build a balanced evaluation subset.')
    parser.add_argument('--data-dir', type=Path, default=Path('/home/tkimura4/data/2024-03-29-ICT'))
    parser.add_argument(
        '--labels-dir',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-11/labels'),
    )
    parser.add_argument(
        '--continuity-summary',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-13/continuity_layer/continuity_per_run_summary.csv'),
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory'),
    )
    parser.add_argument('--target-runs-per-label', type=int, default=3)
    parser.add_argument('--ambiguous-threshold', type=float, default=0.03)
    return parser.parse_args()


def find_all_loop_bounds(axis_m: pd.Series) -> list[tuple[int, int]]:
    smooth = axis_m.rolling(window=5, center=True, min_periods=1).mean().reset_index(drop=True)
    values = smooth.to_list()
    minima: list[int] = []
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


def summarize_run_labels(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            'n_samples': 0,
            'duration_s': 0.0,
            'labeling_usable': False,
            'loop_extraction_usable': False,
            'usable_loop_segments': 0,
            'median_loop_length_s': 0.0,
            'min_loop_length_s': 0.0,
            'max_loop_length_s': 0.0,
            'loop_length_cv': 0.0,
        }

    df = pd.read_csv(path)
    if df.empty:
        return {
            'n_samples': 0,
            'duration_s': 0.0,
            'labeling_usable': False,
            'loop_extraction_usable': False,
            'usable_loop_segments': 0,
            'median_loop_length_s': 0.0,
            'min_loop_length_s': 0.0,
            'max_loop_length_s': 0.0,
            'loop_length_cv': 0.0,
        }

    loops = find_all_loop_bounds(pd.to_numeric(df['axis_m'], errors='coerce'))
    elapsed = pd.to_numeric(df['elapsed_s'], errors='coerce')
    loop_lengths = [float(elapsed.iloc[end] - elapsed.iloc[start]) for start, end in loops]
    median_loop = float(pd.Series(loop_lengths).median()) if loop_lengths else 0.0
    mean_loop = float(pd.Series(loop_lengths).mean()) if loop_lengths else 0.0
    std_loop = float(pd.Series(loop_lengths).std(ddof=0)) if loop_lengths else 0.0
    loop_cv = std_loop / mean_loop if mean_loop > 0 else 0.0

    return {
        'n_samples': int(len(df)),
        'duration_s': float(elapsed.iloc[-1] - elapsed.iloc[0]) if len(df) > 1 else 0.0,
        'labeling_usable': True,
        'loop_extraction_usable': bool(loop_lengths),
        'usable_loop_segments': int(len(loop_lengths)),
        'median_loop_length_s': median_loop,
        'min_loop_length_s': float(min(loop_lengths)) if loop_lengths else 0.0,
        'max_loop_length_s': float(max(loop_lengths)) if loop_lengths else 0.0,
        'loop_length_cv': float(loop_cv),
    }


def build_quality_issue(row: pd.Series, ambiguous_threshold: float, weaker_runs: set[int]) -> str:
    issues: list[str] = []
    if not bool(row['labeling_usable']):
        issues.append('missing_labels')
    if bool(row['labeling_usable']) and not bool(row['loop_extraction_usable']):
        issues.append('no_full_loops')
    if float(row.get('ambiguous_fraction', 0.0)) >= ambiguous_threshold:
        issues.append('high_ambiguous_fraction')
    median_loop = float(row.get('median_loop_length_s', 0.0))
    max_loop = float(row.get('max_loop_length_s', 0.0))
    min_loop = float(row.get('min_loop_length_s', 0.0))
    if median_loop > 0.0 and (max_loop >= 1.5 * median_loop or min_loop <= 0.75 * median_loop):
        issues.append('irregular_loop_lengths')
    if int(row['run_id']) in weaker_runs:
        issues.append('weaker_runtime_continuity')
    return ';'.join(issues)


def selection_reason(row: pd.Series) -> str:
    return (
        f"usable loops={int(row['usable_loop_segments'])}, "
        f"loop_cv={float(row['loop_length_cv']):.3f}, "
        f"runtime_step={float(row['runtime_mean_step_m']):.2f} m"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = pd.read_parquet(args.data_dir / 'run_ids.parquet').copy().sort_values('run_id').reset_index(drop=True)
    label_summary = pd.read_csv(args.labels_dir / 'discrete_label_summary.csv').copy()
    runtime = pd.read_csv(args.continuity_summary).copy()
    runtime = runtime[runtime['tracker_mode'] == 'fixedlag5_smoothed_hybrid_mic'].copy()
    runtime = runtime.rename(
        columns={
            'mean_xy_error_m': 'runtime_mean_xy_error_m',
            'mean_step_m': 'runtime_mean_step_m',
            'joint_station_side_accuracy': 'runtime_joint_station_side_accuracy',
        }
    )
    weaker_runs = set(
        runtime.sort_values(['runtime_mean_xy_error_m', 'runtime_mean_step_m'], ascending=[False, False])['run_id'].head(2).tolist()
    )

    rows: list[dict[str, object]] = []
    for meta in run_meta.itertuples(index=False):
        run_id = int(meta.run_id)
        label_path = args.labels_dir / f'run{run_id}_discrete_labels.csv'
        loop_summary = summarize_run_labels(label_path)
        label_row = label_summary[label_summary['run_id'] == run_id]
        runtime_row = runtime[runtime['run_id'] == run_id]
        row = {
            'run_id': run_id,
            'label': str(meta.label),
            'split': str(meta.set) if hasattr(meta, 'set') else '',
            **loop_summary,
            'ambiguous_fraction': float(label_row['ambiguous_fraction'].iloc[0]) if not label_row.empty else 0.0,
            'median_station_margin_m': float(label_row['median_station_margin_m'].iloc[0]) if not label_row.empty else 0.0,
            'median_sensor_margin_m': float(label_row['median_sensor_margin_m'].iloc[0]) if not label_row.empty else 0.0,
            'runtime_mean_xy_error_m': (
                float(runtime_row['runtime_mean_xy_error_m'].iloc[0]) if not runtime_row.empty else float('nan')
            ),
            'runtime_mean_step_m': float(runtime_row['runtime_mean_step_m'].iloc[0]) if not runtime_row.empty else float('nan'),
            'runtime_joint_station_side_accuracy': (
                float(runtime_row['runtime_joint_station_side_accuracy'].iloc[0]) if not runtime_row.empty else float('nan')
            ),
        }
        rows.append(row)

    inventory = pd.DataFrame(rows).sort_values('run_id').reset_index(drop=True)
    inventory['quality_issue'] = inventory.apply(build_quality_issue, axis=1, ambiguous_threshold=args.ambiguous_threshold, weaker_runs=weaker_runs)
    inventory.to_csv(args.out_dir / 'run_inventory.csv', index=False)

    usable = inventory[inventory['labeling_usable'] & inventory['loop_extraction_usable']].copy()
    usable = usable.sort_values(
        ['label', 'usable_loop_segments', 'loop_length_cv', 'runtime_mean_step_m', 'run_id'],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)
    usable['selection_rank_within_label'] = usable.groupby('label').cumcount() + 1
    usable['selected'] = usable['selection_rank_within_label'] <= args.target_runs_per_label
    usable['selection_reason'] = usable.apply(selection_reason, axis=1)

    subset_rows: list[dict[str, object]] = []
    for label, group in usable.groupby('label', sort=True):
        selected = group[group['selected']].copy()
        shortfall_note = ''
        if len(selected) < args.target_runs_per_label:
            shortfall_note = f'only {len(selected)} usable runs available for target {label}; target was {args.target_runs_per_label}'
        for row in selected.itertuples(index=False):
            subset_rows.append(
                {
                    'run_id': int(row.run_id),
                    'label': str(row.label),
                    'selected': True,
                    'selection_rank_within_label': int(row.selection_rank_within_label),
                    'selection_reason': str(row.selection_reason),
                    'shortfall_note': shortfall_note,
                }
            )

    subset = pd.DataFrame(subset_rows).sort_values(['label', 'selection_rank_within_label', 'run_id']).reset_index(drop=True)
    subset.to_csv(args.out_dir / 'balanced_subset.csv', index=False)


if __name__ == '__main__':
    main()
