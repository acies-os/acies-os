"""Convert mic SQLite DB directly to WAV, ignoring timestamps.

Reads rows in timestamp order, concatenates payloads into a single array,
and writes a WAV file. Timestamps are ignored -- gaps in the DB will appear
as missing audio rather than silence.

Usage::

    python db_to_wav.py /data/rs1-mic.db
    python db_to_wav.py /data/rs1-mic.db -o out.wav
    python db_to_wav.py /data/rs1-mic.db --topic rs1/mic
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click
import msgspec.json
import numpy as np
import scipy.io.wavfile


@click.command()
@click.argument('db_path', metavar='DB', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-o', '--output', type=click.Path(dir_okay=False), default=None, help='Output WAV path. Defaults to <db>.wav.'
)
@click.option('--topic', default=None, help='Filter by topic.')
def main(db_path: str, output: str | None, topic: str | None) -> None:
    """Concatenate DB payloads into a WAV file, ignoring timestamps."""
    src = Path(db_path)
    dst = Path(output) if output else src.with_suffix('.wav')

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
        print('no rows found', file=sys.stderr)
        return

    dtype, _, meta_bytes = rows[0]
    meta: dict = msgspec.json.decode(meta_bytes)
    sample_rate: int = meta['sampling_rate']
    print(f'rows:        {len(rows)}', file=sys.stderr)
    print(f'dtype:       {dtype}', file=sys.stderr)
    print(f'sample rate: {sample_rate} Hz', file=sys.stderr)

    samples = np.concatenate([np.array(msgspec.json.decode(payload), dtype=dtype) for _, payload, _ in rows])
    print(f'duration:    {len(samples) / sample_rate:.1f}s', file=sys.stderr)

    scipy.io.wavfile.write(dst, sample_rate, samples)
    print(f'saved {dst}', file=sys.stderr)


if __name__ == '__main__':
    main()
