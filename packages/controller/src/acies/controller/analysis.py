import logging
from collections import defaultdict
from dataclasses import dataclass

from acies.controller.ns import make_ctl_topic, parse_node_name, parse_service_name

logger = logging.getLogger('acies.controller')


@dataclass
class ServiceState:
    id: int
    kind: str
    node: str
    service: str
    timestamp_ns: int
    state: dict

    def is_running(self) -> bool:
        deactivated = self.state.get('deactivated', False)
        return not deactivated

    def is_noisy(self, energy: dict, threshold: dict) -> bool:
        return energy[self.service] > threshold[self.node][self.service]


def group_states_by_node(states: list[ServiceState]) -> dict[str, list[ServiceState]]:
    """Assumption:
    - Normal service: `<node>/<service>/ctl`
    - Backup service: `<hosting_node>/backup/<node>/<service>/ctl`

    A backup service belongs to `<node>` instead of `<hosting_node>`.
    """
    node_states = defaultdict(list)

    for s in states:
        if s.service.startswith('backup/'):
            node = parse_node_name(s.service)
        else:
            node = s.node
        node_states[node].append(s)
    # convert to normal dict and sort the list by id
    node_states = {k: sorted(v, key=lambda x: x.id) for k, v in node_states.items()}
    return node_states


def failover_check_node(
    reply_to: str,
    node: str,
    states: list[ServiceState],
    deps: dict[str, set[str]],
) -> list[dict]:
    live = {x.service: x for x in states if x.is_running()}
    backup = {x.service: x for x in states if not x.is_running()}

    dep_to_backup = defaultdict(list)
    for x in states:
        dep_key = parse_service_name(x.service)
        dep_to_backup[dep_key].append(x)

    # # sort the backup services in the order in `deps` so that we can user order of keys in `deps` to show preference
    # # here we use a list of tuples because dep_key:backup_key could be a 1:n mapping
    # dep_to_backup = [
    #     (dep_key, backup_key) for backup_key in backup for dep_key in deps if parse_service_name(backup_key) == dep_key
    # ]
    # dep_to_backup = sorted(dep_to_backup, key=lambda x: list(deps.keys()).index(x[0]))
    # backup_sorted = {backup_key: backup[backup_key] for _, backup_key in dep_to_backup}
    #
    # backup_to_dep = {v: k for k, v in dep_to_backup}

    changes = []
    n_healthy = 0

    def preference_sort(xs: dict):
        for x in xs:
            if not x.startswith('backup/'):
                return x
        return next(iter(xs))

    current_view = defaultdict(dict)
    for x in states:
        k = parse_service_name(x.service)
        if k in deps:
            current_view[k][x.service] = x

    # current_view = {
    #     'vfm': {},
    #     'vfm-geo': {},
    #     'vfm-mic': {},
    # }

    # default: everything deactivated
    expect_view = {k: {xs: False} for k in current_view for xs in current_view[k]}
    for k, v in deps.items():
        if k in expect_view and all(d in live for d in v):
            x = preference_sort(expect_view[k])
            expect_view[k][x] = True
            break

    for k in expect_view:
        for sk in expect_view[k]:
            if current_view[k][sk].is_running() != expect_view[k][sk]:
                topic_to = make_ctl_topic(current_view[k][sk].node, current_view[k][sk].service)
                changes.append({'topic': topic_to, 'payload': {'deactivated': not expect_view[k][sk]}})

    return changes


def noise_check_node(
    states_liveness: list[ServiceState],
    states_noiseness: list[ServiceState],
    noise_thresh: dict[str, dict[str, int]],
) -> list[ServiceState]:
    logger.debug('----------------------------------------------------')
    # get sensor states for geo and mic services
    sensor_states = [x for x in states_liveness if x.service in ['geo', 'mic']]
    # get remaining states
    other_states = [x for x in states_liveness if x.service not in ['geo', 'mic', 'noise-detector']]

    # filter sensor states based on energy level from noise detector service
    mod_energy_level = states_noiseness[0].state

    clean_sensor_states = [x for x in sensor_states if mod_energy_level[x.service] < noise_thresh[x.node][x.service]]
    result = clean_sensor_states + other_states

    # return failover_check_node(reply_to, node, clean_sensor_states + other_states, deps)
    return result
