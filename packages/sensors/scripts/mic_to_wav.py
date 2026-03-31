# pyright: strict, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Convert mic data (SQLite DB or acies-dump CSV) to WAV.

DB mode  -- reads rows in timestamp order, concatenates JSON payloads;
            sample rate is taken from the first row's metadata.
            Gaps appear as missing audio, not silence.

CSV mode -- reads `value` column (int16 PCM), infers sample rate from
            median timestamp delta.

Usage::

    python mic_to_wav.py input.db
    python mic_to_wav.py input.db --topic rs1/mic
    python mic_to_wav.py input.csv
    python mic_to_wav.py input.csv -o out.wav
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click
import msgspec.json
import numpy as np
import polars as pl
import scipy.io.wavfile


def _from_db(src: Path, topic: str | None) -> tuple[np.ndarray[tuple[int], np.dtype[np.int16]], int]:
    uri = f'file:{src.expanduser()}?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    try:
        if topic:
            rows = con.execute(
                'SELECT dtype, payload, metadata FROM message WHERE topic = ? ORDER BY timestamp',
                (topic,),
            ).fetchall()
        else:
            rows = con.execute('SELECT dtype, payload, metadata FROM message ORDER BY timestamp').fetchall()
    finally:
        con.close()

    if not rows:
        raise click.ClickException('no rows found in DB')

    dtype, _, meta_bytes = rows[0]
    meta: dict[str, object] = msgspec.json.decode(meta_bytes)
    sample_rate = int(meta['sampling_rate'])  # pyright: ignore[reportArgumentType]
    print(f'rows:        {len(rows)}', file=sys.stderr)
    print(f'dtype:       {dtype}', file=sys.stderr)
    print(f'sample rate: {sample_rate} Hz', file=sys.stderr)

    samples = np.concatenate([np.array(msgspec.json.decode(payload), dtype=dtype) for _, payload, _ in rows])
    return samples.astype(np.int16), sample_rate


def _from_csv(src: Path) -> tuple[np.ndarray[tuple[int], np.dtype[np.int16]], int]:
    df = pl.read_csv(src).sort('timestamp_ns')
    ts = df['timestamp_ns'].to_numpy()
    median_delta_ns = int(np.median(np.diff(ts)))
    sample_rate = round(1_000_000_000 / median_delta_ns)
    print(f'inferred sample rate: {sample_rate} Hz', file=sys.stderr)
    return df['value'].to_numpy().astype(np.int16), sample_rate


@click.command()
@click.argument('input_path', metavar='FILE', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-o', '--output', type=click.Path(dir_okay=False), default=None, help='Output WAV path. Defaults to <input>.wav.'
)
@click.option('--topic', default=None, help='Filter by Zenoh topic (DB mode only).')
def main(input_path: str, output: str | None, topic: str | None) -> None:
    """Convert mic DB or CSV data to WAV."""
    src = Path(input_path)
    dst = Path(output) if output else src.with_suffix('.wav')
    ext = src.suffix.lower()

    if ext == '.db':
        samples, sample_rate = _from_db(src, topic)
    elif ext == '.csv':
        if topic:
            raise click.UsageError('--topic is only supported for DB files')
        samples, sample_rate = _from_csv(src)
    else:
        raise click.BadParameter(f'unsupported file type: {ext!r} (expected .db or .csv)')

    print(f'duration:    {len(samples) / sample_rate:.1f}s', file=sys.stderr)
    _ = scipy.io.wavfile.write(dst, sample_rate, samples)  # pyright: ignore[reportArgumentType]
    print(f'saved {dst}', file=sys.stderr)


if __name__ == '__main__':
    main()
