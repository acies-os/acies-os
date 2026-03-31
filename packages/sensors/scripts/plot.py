# pyright: strict, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Plot sensor data (CSV or Parquet) as a time series JPEG.

Usage::

    python plot.py input.csv
    python plot.py input.parquet
    python plot.py input.csv -o out.jpg

CSV schema:  timestamp_ns (Int64), channel (String), value (Int64/Float64)
Parquet schema: timestamp (Float64, Unix seconds), samples (Int64),
                channel (String), original_timestamp (Float64)
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import polars as pl


def _load(src: Path) -> pl.DataFrame:
    """Load file and return a DataFrame with columns: time (Datetime), channel, value.

    Handles two CSV and four Parquet schema variants found in this project:
    - CSV: timestamp_ns (Int64 ns), channel (str), value
    - Parquet 2024 geo/mic: timestamp (Float64 s), channel (str or int), samples
    - Parquet 2023 geo: timestamp (Int64 ms), channel (str), samples
    - Parquet 2023 mic: timestamp (Int64 ms), no channel column, samples
    """
    ext = src.suffix.lower()
    if ext == '.csv':
        df = pl.read_csv(src).with_columns(pl.from_epoch('timestamp_ns', time_unit='ns').alias('time'))
        return df.select('time', 'channel', pl.col('value'))
    elif ext == '.parquet':
        df = pl.read_parquet(src)

        # --- normalize timestamp to Datetime ---
        ts = df['timestamp']
        if ts.dtype == pl.Float64:
            # Float64 seconds -> convert to ns Int64
            time_col = pl.from_epoch(
                (pl.col('timestamp') * 1_000_000_000).cast(pl.Int64),
                time_unit='ns',
            ).alias('time')
        elif ts.dtype == pl.Int64 and ts[0] > 1_000_000_000_000:
            # 13-digit Int64 -> milliseconds
            time_col = pl.from_epoch('timestamp', time_unit='ms').alias('time')
        else:
            # 10-digit Int64 -> seconds
            time_col = pl.from_epoch('timestamp', time_unit='s').alias('time')

        df = df.with_columns(time_col)

        # --- normalize channel ---
        if 'channel' in df.columns:
            df = df.with_columns(pl.col('channel').cast(pl.String))
        else:
            # no channel column: label with filename stem
            df = df.with_columns(pl.lit(src.stem).alias('channel'))

        return df.select('time', 'channel', pl.col('samples').alias('value'))
    else:
        raise click.BadParameter(f'unsupported file type: {ext!r} (expected .csv or .parquet)')


@click.command()
@click.argument('input_path', metavar='FILE', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-o', '--output', type=click.Path(dir_okay=False), default=None, help='Output JPEG path. Defaults to <input>.jpg.'
)
def main(input_path: str, output: str | None) -> None:
    """Plot sensor CSV or Parquet data as a time series JPEG."""
    src = Path(input_path)
    dst = Path(output) if output else src.with_suffix('.jpg')

    df = _load(src)

    channels = sorted(df['channel'].unique().to_list())
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3 * len(channels)), squeeze=False)

    for ax, channel in zip(axes[:, 0], channels):
        subset = df.filter(pl.col('channel') == channel)
        ax.plot(subset['time'].to_numpy(), subset['value'].to_numpy(), linewidth=0.5)
        ax.set_title(f'channel {channel}')
        ax.set_xlabel('time')
        ax.set_ylabel('value')

    _ = fig.suptitle(str(src))
    fig.tight_layout()
    _ = fig.savefig(dst, format='jpeg', dpi=150)
    print(f'saved {dst}', file=sys.stderr)


if __name__ == '__main__':
    main()
