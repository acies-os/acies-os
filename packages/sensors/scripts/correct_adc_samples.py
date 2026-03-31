"""Reprocess a parquet file column from Rust 32-bit signed to Python 24-bit signed.

The Rust parser interprets geophone hex strings as u32 cast to i32 (full 32-bit
two's complement). The Python parser masks to 24 bits first, then applies 24-bit
sign, which matches the device's actual 24-bit ADC output.

Conversion:
    rust_i32  ->  n & 0xFFFFFF  ->  subtract 2^24 if >= 2^23

After the ADC correction, applies standard seismic preprocessing:
    1. Detrend  - remove linear trend
    2. Demean   - remove median (robust to outliers vs mean)
"""

from pathlib import Path

import click
import numpy as np
import polars as pl


def rust_i32_to_24bit(col: pl.Expr) -> pl.Expr:
    """Convert a column of Rust i32 samples to correct 24-bit signed integers."""
    masked = col & 0xFFFFFF
    return pl.when(masked >= (1 << 23)).then(masked - (1 << 24)).otherwise(masked)


def _detrend_demean(s: pl.Series) -> pl.Series:
    arr = s.to_numpy().astype(np.float64)
    x = np.arange(len(arr))
    slope, intercept = np.polyfit(x, arr, 1)
    arr -= slope * x + intercept
    arr -= np.median(arr)
    return pl.Series(arr, dtype=pl.Float64)


@click.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path), default=None)
@click.option('--column', '-c', required=True, help='Column name to reprocess.')
def main(input_file: Path, output_file: Path | None, column: str) -> None:
    """Reprocess INPUT_FILE and write corrected parquet to OUTPUT_FILE."""
    df = pl.read_parquet(input_file)

    if column not in df.columns:
        raise click.BadParameter(
            f"column '{column}' not found; available: {df.columns}",
            param_hint='--column',
        )

    # --- ADC correction (int -> int) ---
    df_adc = df.with_columns(rust_i32_to_24bit(pl.col(column)).alias(column))
    changed = (df[column] != df_adc[column]).sum()
    click.echo(f'ADC-corrected rows: {changed} / {len(df)}')

    # --- detrend + demean (int -> float) ---
    df_out = df_adc.with_columns(pl.col(column).map_batches(_detrend_demean, return_dtype=pl.Float64).alias(column))

    if output_file is None:
        output_file = input_file.with_name(input_file.stem + '_corrected.parquet')

    df_out.write_parquet(output_file)
    click.echo(f'Written {len(df_out)} rows to {output_file}')


if __name__ == '__main__':
    main()
