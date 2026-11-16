import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger()

sql_create_table_stream_buffer = """
create table if not exists stream_buffer (
    id integer primary key,
    node_name text not null,
    model_name text not null,
    timestamp integer not null,
    prediction text not null,
    metadata text not null,
    status text default 'unprocessed'
);
"""


def connect_db(db_path: str | Path, create_table_sql: list[str] | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if create_table_sql:
        with conn:
            for sql in create_table_sql:
                conn.execute(sql)
    return conn


class EnsembleBuffer:
    def __init__(self, db_path: str | Path):
        self._conn = connect_db(db_path, [sql_create_table_stream_buffer])
        self._latest = {}
        self._latest_meta = {}

    def add_entry(
        self, node_name: str, model_name: str, timestamp: int, prediction: dict, metadata: dict | None
    ):
        _prediction = json.dumps(prediction)
        _metadata = json.dumps(metadata)
        with self._conn:
            self._conn.execute(
                'insert into stream_buffer (node_name, model_name, timestamp, prediction, metadata) values (?, ?, ?, ?, ?)',
                (node_name, model_name, timestamp, _prediction, _metadata),
            )
        self._latest[(node_name, model_name)] = (timestamp, prediction)
        self._latest_meta[(node_name, model_name)] = metadata

    def get_latest(self, node_name: str, model_name: str) -> tuple[int, dict]:
        return self._latest[(node_name, model_name)]

    def get_latest_meta(self, node_name: str, model_name: str) -> dict | None:
        return self._latest_meta[(node_name, model_name)]

    def count(self) -> dict:
        with self._conn:
            c = self._conn.execute(
                'select status, count(id) as count '
                'from stream_buffer '
                'where status in ("unprocessed", "done") '
                'group by status;'
            ).fetchall()
        return dict(c)

    def get_unprocessed(self, earliest_timestamp: int, latest_timestamp: int) -> list[dict]:
        sql = """select id, node_name, model_name, timestamp
                 from stream_buffer
                 where status = "unprocessed"
                       and timestamp between ? and ?
                 order by timestamp asc;"""
        with self._conn:
            rows = self._conn.execute(sql, (earliest_timestamp, latest_timestamp)).fetchall()
        rows = [dict(row) for row in rows]
        return rows

    def mark_as_done(self, ids: list[int]):
        sql = 'update stream_buffer set status = "done" where id in ({})'.format(','.join('?' * len(ids)))
        params = tuple(ids)
        with self._conn:
            cur = self._conn.execute(sql, params)
            n_rows = cur.rowcount
        return n_rows

    def get_data(self, rows: list[dict], temporal_window_s: int) -> list[dict]:
        t_min = min(row['timestamp'] for row in rows)
        t_max = max(row['timestamp'] for row in rows)
        sql = """select * from stream_buffer where timestamp between ? and ? order by timestamp asc;"""
        param = (t_min - temporal_window_s, t_max + temporal_window_s)
        with self._conn:
            rows = self._conn.execute(sql, param).fetchall()
        return [dict(row) for row in rows]

    def get_range(self, t_min: int, t_max: int) -> list[dict]:
        sql = """select * from stream_buffer where timestamp between ? and ? order by timestamp asc;"""
        param = (t_min, t_max)
        with self._conn:
            rows = self._conn.execute(sql, param).fetchall()
        return [dict(row) for row in rows]
