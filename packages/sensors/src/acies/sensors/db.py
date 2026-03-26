"""Shared SQLite helpers for AciesOS sensor nodes.

All sensor nodes write to a common message schema. Use CAST(... AS TEXT) to
read BLOB columns as JSON strings from the CLI::

    sqlite3 /data/host-geo.db \\
      "SELECT topic, source, dtype, datetime(timestamp/1e9, 'unixepoch'),
              CAST(payload AS TEXT), CAST(metadata AS TEXT)
       FROM message
       WHERE timestamp BETWEEN <start_ns> AND <end_ns>
       ORDER BY timestamp"
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DbRow = tuple[
    str,  # topic
    str,  # numpy dtype as string, e.g. 'int16', 'int32'
    int,  # timestamp in nanoseconds since epoch
    str,  # source, e.g. 'rs1/geo'
    bytes,  # payload, e.g. a numpy array serialized with .tobytes()
    bytes,  # metadata, e.g. a JSON-encoded dict of additional info
]


def open_db(
    path: str,
    check_same_thread: bool = True,
    wal_autocheckpoint: int = 1000,
) -> sqlite3.Connection:
    """Open (or create) the SQLite message database at *path*.

    Creates the message table and index if they do not exist, and configures
    WAL journal mode and NORMAL synchronous for concurrent reads and performance.

    Args:
        check_same_thread: set False when the connection is opened on one thread
            and written from another. Safe as long as only one thread writes.
        wal_autocheckpoint: WAL checkpoint threshold in pages (default 1000 =
            ~4MB). Raise this (e.g. 4000) on I/O-constrained devices like a
            Raspberry Pi to reduce checkpoint frequency at the cost of a larger
            WAL file.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), check_same_thread=check_same_thread)
    _ = con.executescript(f"""
        CREATE TABLE IF NOT EXISTS message (
            id        INTEGER PRIMARY KEY,
            topic     TEXT NOT NULL,
            dtype     TEXT NOT NULL,
            timestamp INT  NOT NULL,
            source    TEXT NOT NULL,
            payload   BLOB NOT NULL,
            metadata  BLOB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_message_timestamp ON message (timestamp);
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA wal_autocheckpoint={wal_autocheckpoint};
    """)
    return con


def flush(con: sqlite3.Connection, rows: list[DbRow]) -> int:
    """Insert a batch of rows, commit, and return the number of rows inserted."""
    cur = con.executemany(
        'INSERT INTO message (topic, dtype, timestamp, source, payload, metadata) VALUES (?,?,?,?,?,?)',
        rows,
    )
    con.commit()
    return cur.rowcount
