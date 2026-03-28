"""Plot sensor CSV data (from acies-dump) as a time series JPEG.

Usage::

    python plot_csv.py input.csv
    python plot_csv.py input.csv -o out.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl


@click.command()
@click.argument('input_path', metavar='CSV', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-o', '--output', type=click.Path(dir_okay=False), default=None, help='Output JPEG path. Defaults to <input>.jpg.'
)
def main(input_path: str, output: str | None) -> None:
    """Plot acies-dump CSV as a time series JPEG."""
    src = Path(input_path)
    dst = Path(output) if output else src.with_suffix('.jpg')

    df = pl.read_csv(src).with_columns(pl.from_epoch('timestamp_ns', time_unit='ns').alias('time'))

    channels = df['channel'].unique().to_list()
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3 * len(channels)), squeeze=False)

    for ax, channel in zip(axes[:, 0], channels):
        subset = df.filter(pl.col('channel') == channel)
        ax.plot(subset['time'].to_numpy(), subset['value'].to_numpy(), linewidth=0.5)
        ax.set_title(f'channel {channel}')
        ax.set_xlabel('time')
        ax.set_ylabel('value')

    fig.suptitle(str(src))
    fig.tight_layout()
    fig.savefig(dst, format='jpeg', dpi=150)
    print(f'saved {dst}', file=sys.stderr)


if __name__ == '__main__':
    main()
