"""Check timestamp continuity in a sensor SQLite DB.

Plots the gap between consecutive row timestamps to make missing rows obvious.
Each row should be exactly 1 second apart; gaps larger than that indicate
missing data.

Usage::

    python check_timestamps.py /data/rs1-mic.db
    python check_timestamps.py /data/rs1-mic.db -o gaps.jpg
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np


@click.command()
@click.argument('db_path', metavar='DB', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option(
    '-o', '--output', type=click.Path(dir_okay=False), default=None, help='Output JPEG path. Defaults to <db>.jpg.'
)
@click.option('--topic', default=None, help='Filter by topic.')
def main(db_path: str, output: str | None, topic: str | None) -> None:
    """Plot timestamp gaps in a sensor SQLite DB."""
    src = Path(db_path)
    dst = Path(output) if output else src.with_suffix('.jpg')

    uri = f'file:{src.expanduser()}?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    try:
        if topic:
            rows = con.execute('SELECT timestamp FROM message WHERE topic = ? ORDER BY timestamp', (topic,)).fetchall()
        else:
            rows = con.execute('SELECT timestamp FROM message ORDER BY timestamp').fetchall()
    finally:
        con.close()

    if not rows:
        print('no rows found', file=sys.stderr)
        return

    ts = np.array([r[0] for r in rows], dtype=np.int64)
    t_s = (ts - ts[0]) / 1e9  # seconds from start
    diff_s = np.diff(ts) / 1e9  # gap between consecutive rows in seconds

    expected_s = np.median(diff_s)
    n_gaps = int(np.sum(diff_s > expected_s * 1.5))
    print(f'rows:          {len(ts)}', file=sys.stderr)
    print(f'duration:      {t_s[-1]:.1f}s', file=sys.stderr)
    print(f'expected gap:  {expected_s:.3f}s', file=sys.stderr)
    print(f'missing rows:  {n_gaps}', file=sys.stderr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

    # --- top: gap between consecutive rows ---
    ax1.plot(t_s[1:], diff_s, linewidth=0.7)
    ax1.axhline(expected_s, color='green', linewidth=0.8, linestyle='--', label=f'expected ({expected_s:.3f}s)')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('gap (s)')
    ax1.set_title('gap between consecutive rows (spikes = missing data)')
    ax1.legend()

    # --- bottom: cumulative drift (actual vs expected timeline) ---
    expected_ts = np.arange(len(ts)) * expected_s
    drift_s = t_s - expected_ts
    ax2.plot(t_s, drift_s, linewidth=0.7)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('drift (s)')
    ax2.set_title('cumulative drift from expected timeline (steps = gaps)')

    fig.suptitle(str(src))
    fig.tight_layout()
    fig.savefig(dst, format='jpeg', dpi=150)
    print(f'saved {dst}', file=sys.stderr)


if __name__ == '__main__':
    main()
