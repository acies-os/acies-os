#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare ICT pooled vs target-specific template scopes.')
    parser.add_argument(
        '--inventory-csv',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/run_inventory.csv'),
    )
    parser.add_argument(
        '--subset-csv',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-13_balanced_eval/inventory/balanced_subset.csv'),
    )
    parser.add_argument(
        '--labels-dir',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-11/labels'),
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('docs/design/artifacts/ict_tracker_2026-04-13_template_scope_eval'),
    )
    return parser.parse_args()


def _selected_runs(subset_csv: Path) -> list[int]:
    subset_df = pd.read_csv(subset_csv).copy()
    if 'selected' in subset_df.columns:
        selected = subset_df[subset_df['selected'].astype(bool)].copy()
    else:
        selected = subset_df.copy()
    return sorted(selected['run_id'].astype(int).unique().tolist())


def _build_manifest_rows(run_meta: pd.DataFrame, selected_runs: list[int]) -> pd.DataFrame:
    meta = run_meta[run_meta['run_id'].isin(selected_runs)].copy()
    rows: list[dict[str, object]] = []
    experiments = ['pooled_loro', 'cross_target_pooled', 'target_specific_paired']
    for eval_row in meta.itertuples(index=False):
        eval_run_id = int(eval_row.run_id)
        eval_label = str(eval_row.label)
        for experiment_name in experiments:
            if experiment_name == 'pooled_loro':
                train_df = meta[meta['run_id'] != eval_run_id].copy()
            elif experiment_name == 'cross_target_pooled':
                train_df = meta[(meta['run_id'] != eval_run_id) & (meta['label'] != eval_label)].copy()
            else:
                train_df = meta[(meta['run_id'] != eval_run_id) & (meta['label'] == eval_label)].copy()
            for train_row in train_df.itertuples(index=False):
                rows.append(
                    {
                        'experiment_name': experiment_name,
                        'eval_run_id': eval_run_id,
                        'eval_label': eval_label,
                        'train_run_id': int(train_row.run_id),
                        'train_label': str(train_row.label),
                        'scope_name': experiment_name,
                    }
                )
    return pd.DataFrame(rows).sort_values(['experiment_name', 'eval_run_id', 'train_run_id']).reset_index(drop=True)


def _run_command(cmd: list[str], workdir: Path) -> None:
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def _collect_outputs(root: Path, experiments: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    per_run_frames = []
    for experiment_name in experiments:
        cont_dir = root / experiment_name / 'continuity_layer'
        summary_df = pd.read_csv(cont_dir / 'continuity_summary.csv').copy()
        per_run_df = pd.read_csv(cont_dir / 'continuity_per_run_summary.csv').copy()
        if 'experiment_name' not in summary_df.columns:
            summary_df['experiment_name'] = experiment_name
        if 'scope_name' not in summary_df.columns:
            summary_df['scope_name'] = experiment_name
        if 'experiment_name' not in per_run_df.columns:
            per_run_df['experiment_name'] = experiment_name
        if 'scope_name' not in per_run_df.columns:
            per_run_df['scope_name'] = experiment_name
        summary_frames.append(summary_df)
        per_run_frames.append(per_run_df)

    summary_all = pd.concat(summary_frames, ignore_index=True)
    per_run_all = pd.concat(per_run_frames, ignore_index=True)
    wanted_modes = ['fixedlag5_smoothed_hybrid_mic', 'smoothed_hybrid_mic']
    summary_keep = summary_all[summary_all['tracker_mode'].isin(wanted_modes)].copy()
    per_run_keep = per_run_all[per_run_all['tracker_mode'].isin(wanted_modes)].copy()
    by_label = (
        per_run_keep.groupby(['experiment_name', 'scope_name', 'tracker_mode', 'label'], as_index=False)[
            ['station_accuracy', 'side_accuracy', 'joint_station_side_accuracy', 'mean_xy_error_m', 'p95_xy_error_m', 'mean_step_m', 'reset_rate']
        ]
        .mean()
        .sort_values(['tracker_mode', 'experiment_name', 'label'])
        .reset_index(drop=True)
    )
    return summary_keep, per_run_keep, by_label


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory_df = pd.read_csv(args.inventory_csv).copy()
    selected_runs = _selected_runs(args.subset_csv)
    run_meta = inventory_df[['run_id', 'label']].drop_duplicates().copy()
    run_meta['run_id'] = run_meta['run_id'].astype(int)
    run_meta = run_meta[run_meta['run_id'].isin(selected_runs)].sort_values('run_id').reset_index(drop=True)

    manifest_df = _build_manifest_rows(run_meta, selected_runs)
    manifest_path = args.out_dir / 'template_scope_train_pairs.csv'
    manifest_df.to_csv(manifest_path, index=False)

    run_args = [str(run_id) for run_id in selected_runs]
    experiments = ['pooled_loro', 'cross_target_pooled', 'target_specific_paired']
    tracker_script = Path('packages/controllers/scripts/eval_ict_simple_tracker.py')
    continuity_script = Path('packages/controllers/scripts/eval_ict_continuity_layer.py')

    for experiment_name in experiments:
        exp_root = args.out_dir / experiment_name
        simple_out = exp_root / 'simple_tracker'
        cont_out = exp_root / 'continuity_layer'
        simple_out.mkdir(parents=True, exist_ok=True)
        cont_out.mkdir(parents=True, exist_ok=True)

        _run_command(
            [
                sys.executable,
                str(tracker_script),
                '--labels-dir',
                str(args.labels_dir),
                '--out-dir',
                str(simple_out),
                '--runs',
                *run_args,
                '--train-manifest',
                str(manifest_path),
                '--experiment-name',
                experiment_name,
            ],
            workdir=Path.cwd(),
        )

        _run_command(
            [
                sys.executable,
                str(continuity_script),
                '--predictions',
                str(simple_out / 'tracker_predictions.csv'),
                '--sensor-geometry',
                str(simple_out / 'sensor_geometry.csv'),
                '--out-dir',
                str(cont_out),
                '--base-mode',
                'hybrid_mic',
                '--template-weight',
                '1.8',
                '--anchor-weight',
                '1.0',
                '--neighbor-anchor-weight',
                '0.55',
                '--max-step-nodes',
                '2',
                '--hop-penalty',
                '0.7',
                '--direction-bonus',
                '0.35',
                '--direction-penalty',
                '0.25',
                '--stay-penalty',
                '0.05',
                '--reset-penalty',
                '4.0',
                '--lag-steps',
                '5',
                '--loop-selection',
                'best_smoothed',
                '--runs',
                *run_args,
                '--plot-runs',
                *run_args,
                '--train-manifest',
                str(manifest_path),
                '--experiment-name',
                experiment_name,
            ],
            workdir=Path.cwd(),
        )

    summary_df, per_run_df, by_label_df = _collect_outputs(args.out_dir, experiments)
    summary_df.to_csv(args.out_dir / 'continuity_scope_summary.csv', index=False)
    per_run_df[per_run_df['tracker_mode'] == 'fixedlag5_smoothed_hybrid_mic'].to_csv(
        args.out_dir / 'fixedlag5_scope_per_run.csv',
        index=False,
    )
    by_label_df[by_label_df['tracker_mode'] == 'fixedlag5_smoothed_hybrid_mic'].to_csv(
        args.out_dir / 'fixedlag5_scope_by_label.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
