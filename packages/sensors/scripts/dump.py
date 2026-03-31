# pyright: strict, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Export AciesOS sensor SQLite data to CSV for plotting.

Each DB row holds 1 second of samples. Per-sample timestamps are interpolated
linearly from the row's window-start timestamp::

    sample_ts_ns = row_ts_ns + int(i * 1_000_000_000 / sampling_rate)

Output columns: timestamp_ns, value, channel, source, topic

Usage::

    python dump.py /data/host-geo.db > out.csv
    python dump.py /data/host-geo.db --duration 30 > out.csv
    python dump.py /data/host-geo.db --start 1700000000000000000 --duration 10 > out.csv
    python dump.py /data/host-mic.db --topic rs1/mic > out.csv
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path

import click
import msgspec.json


def _iter_rows(
    con: sqlite3.Connection,
    topic: str | None,
    start: int | None,
    end: int | None,
) -> Iterator[tuple[str, str, int, str, bytes, bytes]]:
    """Yield (topic, dtype, timestamp, source, payload, metadata) from message."""
    clauses: list[str] = []
    params: list[object] = []

    if topic is not None:
        clauses.append('topic = ?')
        params.append(topic)
    if start is not None:
        clauses.append('timestamp >= ?')
        params.append(start)
    if end is not None:
        clauses.append('timestamp <= ?')
        params.append(end)

    where = ('WHERE ' + ' AND '.join(clauses)) if clauses else ''
    sql = f'SELECT topic, dtype, timestamp, source, payload, metadata FROM message {where} ORDER BY timestamp'
    yield from con.execute(sql, params)


def _expand_row(
    topic: str,
    _dtype: str,
    row_ts_ns: int,
    source: str,
    payload: bytes,
    metadata: bytes,
) -> Iterator[tuple[int, int, object, str, str]]:
    """Expand one DB row into (timestamp_ns, value, channel, source, topic) per sample."""
    samples: list[int] = msgspec.json.decode(payload)
    meta: dict = msgspec.json.decode(metadata)  # pyright: ignore[reportMissingTypeArgument]
    sampling_rate = meta['sampling_rate']
    channel = meta['channel']
    ns_per_sample = 1_000_000_000 / sampling_rate

    for i, value in enumerate(samples):
        ts_ns = row_ts_ns + int(i * ns_per_sample)
        yield ts_ns, value, channel, source, topic


@click.command()
@click.argument('input_path', metavar='DB', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.option('--topic', default=None, help='Filter by topic.')
@click.option(
    '--start',
    default=None,
    type=int,
    help='Start timestamp filter (nanoseconds since epoch, inclusive).',
)
@click.option(
    '--end',
    default=None,
    type=int,
    help='End timestamp filter (nanoseconds since epoch, inclusive).',
)
@click.option(
    '--duration',
    default=None,
    type=float,
    help=(
        'Duration in seconds. '
        'With --start: sets end = start + duration. '
        'Alone: returns the last <duration> seconds of data.'
    ),
)
def main(
    input_path: str,
    topic: str | None,
    start: int | None,
    end: int | None,
    duration: float | None,
) -> None:
    """Export sensor SQLite data to CSV on stdout (one row per sample)."""
    uri = f'file:{Path(input_path).expanduser()}?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    try:
        match (duration, start, end):
            case (None, _, _):
                pass
            case (float(), int(), None):
                # start + duration -> derive end
                end = start + int(duration * 1_000_000_000)
            case (float(), None, int()):
                # end - duration -> derive start
                start = end - int(duration * 1_000_000_000)
            case (float(), None, None):
                # duration only -> last <duration> seconds
                row = con.execute('SELECT MAX(timestamp) FROM message').fetchone()
                if row[0] is not None:
                    end = row[0]
                    assert isinstance(end, int)
                    start = end - int(duration * 1_000_000_000)
            case _:
                raise click.UsageError('--duration cannot be used together with both --start and --end.')

        writer = csv.writer(sys.stdout)
        writer.writerow(['timestamp_ns', 'value', 'channel', 'source', 'topic'])
        for db_row in _iter_rows(con, topic, start, end):
            for sample_row in _expand_row(*db_row):
                writer.writerow(sample_row)
    finally:
        con.close()


if __name__ == '__main__':
    main()
