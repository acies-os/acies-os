"""Convert mic acies-dump CSV to WAV.

Reads the `value` column (int16 PCM samples) and infers the sampling rate
from timestamp deltas.

Usage::

    python mic_csv_to_wav.py input.csv
    python mic_csv_to_wav.py input.csv -o out.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import polars as pl
import scipy.io.wavfile


@click.command()
@click.argument('input_path', metavar='CSV', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-o', '--output', type=click.Path(dir_okay=False), default=None, help='Output WAV path. Defaults to <input>.wav.'
)
def main(input_path: str, output: str | None) -> None:
    """Convert mic acies-dump CSV to WAV."""
    src = Path(input_path)
    dst = Path(output) if output else src.with_suffix('.wav')

    df = pl.read_csv(src).sort('timestamp_ns')

    ts = df['timestamp_ns'].to_numpy()
    median_delta_ns = int(np.median(np.diff(ts)))
    sample_rate = round(1_000_000_000 / median_delta_ns)
    print(f'inferred sample rate: {sample_rate} Hz', file=sys.stderr)

    samples = df['value'].to_numpy().astype(np.int16)
    scipy.io.wavfile.write(dst, sample_rate, samples)
    print(f'saved {dst}', file=sys.stderr)


if __name__ == '__main__':
    main()
