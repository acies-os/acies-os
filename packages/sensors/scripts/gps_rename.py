"""Rename GPS parquet files using labels from a run_ids parquet.

Extracts the run_id from the GPS filename (e.g. run2_gps.parquet -> 2),
looks it up in the run_ids table, and renames to run<id>_<label>_gps.parquet.

Usage::

    python gps_rename.py run_ids.parquet run2_gps.parquet run3_gps.parquet
    python gps_rename.py run_ids.parquet run*_gps.parquet --dry-run
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import polars as pl

_GPS_RE = re.compile(r'^run(\d+)(?:_.+)?_gps\.parquet$')


@click.command()
@click.argument('run_ids', type=click.Path(exists=True, dir_okay=False))
@click.argument('files', nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--dry-run', is_flag=True, help='Show renames without executing.')
def main(run_ids: str, files: tuple[str, ...], dry_run: bool) -> None:
    """Rename GPS parquet files using labels from RUN_IDS."""
    df = pl.read_parquet(run_ids)
    label_map: dict[int, str] = dict(
        zip(
            df['run_id'].to_list(),
            df['label'].to_list(),
        )
    )

    for filepath in files:
        path = Path(filepath)
        m = _GPS_RE.match(path.name)
        if not m:
            click.echo(f'skip: {path.name} (does not match run<id>_gps.parquet pattern)')
            continue

        run_id = int(m.group(1))
        label = label_map.get(run_id)
        if label is None:
            click.echo(f'skip: {path.name} (run_id={run_id} not found in run_ids)')
            continue

        new_name = f'run{run_id}_{label}_gps.parquet'
        if path.name == new_name:
            click.echo(f'skip: {path.name} (already named correctly)')
            continue

        new_path = path.parent / new_name
        if new_path.exists():
            click.echo(f'skip: {path.name} -> {new_name} (target already exists)')
            continue

        if dry_run:
            click.echo(f'would rename: {path.name} -> {new_name}')
        else:
            _ = path.rename(new_path)
            click.echo(f'renamed: {path.name} -> {new_name}')


if __name__ == '__main__':
    main()
