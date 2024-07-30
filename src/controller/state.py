import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger('acies.controller')


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
