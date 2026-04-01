import json
import logging
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger('acies.controller')


class StateDb:
    def __init__(self):
        query = """create table if not exists system_state (
                     id integer primary key,
                     kind text not null,
                     node text not null,
                     service text not null,
                     timestamp_ns integer not null,
                     state text not null
                     )"""
        self.conn = sqlite3.connect('ctl.db')
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute(query)

    def insert_state(
        self,
        kind: str,
        node_name: str,
        service_name: str,
        timestamp_ns: int,
        state: dict,
    ) -> int:
        state_str = json.dumps(state)
        with self.conn:
            cur = self.conn.execute(
                """insert into system_state (kind, node, service, timestamp_ns, state)
                   values (?, ?, ?, ?, ?)""",
                (kind, node_name, service_name, timestamp_ns, state_str),
            )
            n_rows = cur.rowcount
            return n_rows

    def get_states(self, kind: str, start_ns: int, end_ns: int) -> list[dict]:
        """Get all states of `kind` from timestamp `start_ns` to timestamp `end_ns`

        Example `kind`: heartbeat, noise.
        """
        query = """select * from system_state
                   where kind = ?
                   and timestamp_ns between ? and ?
                   order by timestamp_ns asc"""
        params = [kind, start_ns, end_ns]
        if kind is not None:
            query += ''
        with self.conn:
            rows = self.conn.execute(query, tuple(params))
        rows = [dict(row) for row in rows]
        for row in rows:
            row['state'] = json.loads(row['state'])
        return rows

    def get_latest(self, kind: str, start_ns: int, end_ns: int) -> list[dict]:
        """Get the latest states of `kind` for each (node, service) pair from timestamp `start_ns` to `end_ns`

        Example `kind`: heartbeat, noise.
        """
        # states = self.get_states(kind, start_ns, end_ns)
        # latest = {}
        # for s in states:
        #     k = s['node'], s['service']
        #     if k not in latest:
        #         latest[k] = s
        #     elif s['timestamp_ns'] > latest[k]['timestamp_ns']:
        #         latest[k] = s
        # return list(latest.values())
        query = """
            select t1.*
            from system_state t1
            join (
                select node, service, max(timestamp_ns) as max_timestamp
                from system_state
                where kind == ?
                    and timestamp_ns between ? and ?
                group by node, service
            ) t2
            on t1.node == t2.node
                and t1.service == t2.service
                and t1.timestamp_ns == t2.max_timestamp
            where t1.kind == ? 
                and t1.timestamp_ns between ? and ?
            order by t1.node, t1.service, t1.timestamp_ns desc
        """
        with self.conn:
            rows = self.conn.execute(query, (kind, start_ns, end_ns, kind, start_ns, end_ns))
        rows = [dict(row) for row in rows]
        for row in rows:
            row['state'] = json.loads(row['state'])
        return rows


@dataclass
class SystemStateRecord:
    node_name: str = field(repr=True)
    service_name: str = field(repr=True)
    timestamp: float = field(repr=True)
    system_info: dict = field(repr=False)
    deactivated: bool = field(repr=True)


@dataclass
class SystemStates:
    n_init_services: int

    # current system state
    states: dict[tuple[str, str], SystemStateRecord] = field(default_factory=dict)

    # system state history (DB)
    history: deque[SystemStateRecord] = field(default_factory=lambda: deque(maxlen=360), repr=False)

    initialized: bool = False

    def add_record(
        self,
        node_name: str,
        service_name: str,
        timestamp: float,
        system_info: dict,
        deactivated: bool,
    ):
        if not self.initialized and len(self.states) >= self.n_init_services:
            self.initialized = True

        prev_state = self.states.get((node_name, service_name))
        if not system_info and prev_state is not None:
            # heartbeat only message, use the previous system info
            system_info = prev_state.system_info

        record = SystemStateRecord(node_name, service_name, timestamp, system_info, deactivated)
        self.history.append(record)
        self.states[(node_name, service_name)] = record

    def prune_outdated(self, oldest_timestamp: float):
        for k in list(self.states):
            # logger.debug(f'{k}: {oldest_timestamp - self.states[k].timestamp}')
            if self.states[k].timestamp < oldest_timestamp:
                logger.debug(f'delete stalled heartbeat: {k}')
                del self.states[k]

    def get_node_service_mapping(self) -> dict[str, list[str]]:
        mapping = defaultdict(list)
        for n, s in self.states:
            mapping[n].append(s)
        return mapping

    def list_live_nodes(self) -> list[str]:
        result = [x.node_name for x in self.states.values() if 'backup' not in x.service_name and not x.deactivated]
        # result = [x.node_name for x in self.states.values() if not x.deactivated]
        return sorted(set(result))

    def list_backup_nodes(self) -> list[str]:
        result = list(set(x.node_name for x in self.states.values()))
        for x in self.states.values():
            if not x.deactivated and x.node_name in result:
                result.remove(x.node_name)
        return result

    def list_node_services(self, node_name: str) -> list[SystemStateRecord]:
        services = [v for v in self.states.values() if v.node_name == node_name and 'backup' not in v.service_name]
        return services

    def list_backup_services(self, node_name: str) -> list[SystemStateRecord]:
        backup_services = [
            v
            for v in self.states.values()
            # topic: rs1/backup/rs2/vfm
            # service name: backup/rs2/vfm
            if node_name + '/' in v.service_name and 'backup' in v.service_name
        ]
        return backup_services
