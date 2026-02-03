import json
import logging
import queue
import random
import re
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime

import click
import numpy as np
import tomli as toml
import websockets.sync.server as ws_server

from acies.buffers import EnsembleBuffer, TemporalBuffer
from acies.controller.analysis import ServiceState, failover_check_node, group_states_by_node, noise_check_node
from acies.controller.ns import make_ctl_topic, parse_node_name, parse_service_name
from acies.controller.state import StateDb, SystemStateRecord, SystemStates
from acies.core import AciesMsg, Service, common_options, get_zconf, init_logger, pretty

logger = logging.getLogger('acies.controller')


def get_node_service_timestamp(msg: AciesMsg):
    node_name, service_name = msg.reply_to.split('/', 1)
    service_name = service_name.removesuffix('/ctl')
    # msg.timestamp is in nanoseconds
    timestamp = int(msg.timestamp / 1e9)
    deactivated = msg.get_metadata().get('deactivated', False)
    return node_name, service_name, timestamp, deactivated


def logits_to_predictions(logit_dict: dict[str, float]) -> tuple[str, float]:
    pred, confidence = max(logit_dict.items(), key=lambda x: x[1])
    return pred, confidence


def filter_by_confidence(pred: dict[str, float], thresh: dict[str, float]) -> dict[str, float]:
    result = {k: v for k, v in pred.items() if v >= thresh[k]}
    return result


def parse_node_model(reply_to: str) -> tuple[str, str]:
    reply_to = reply_to.removesuffix('/ctl')
    node_name, model_name = tuple(reply_to.rsplit('/', 1))
    if 'backup' in node_name:
        _, node_name = tuple(node_name.rsplit('/', 1))
    return node_name, model_name


def soft_vote(pred_list: list[dict[str, float]], size: int) -> dict[str, float]:
    """Multi-label prediction soft voting."""
    if not pred_list:
        return {}
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)
    for pred in pred_list:
        for k, v in pred.items():
            sum_dict[k] += v
            count_dict[k] += 1
    mean_pred = {k: sum_dict[k] / count_dict[k] for k in sum_dict}
    mean_pred = {k: v for k, v in mean_pred.items() if count_dict[k] >= size}
    logger.debug(f'soft_vote: {dict(count_dict)}')
    return mean_pred


def hard_vote(pred_list: list[dict[str, float]]) -> dict[str, float]:
    """Multi-label prediction hard voting."""
    if not pred_list:
        return {}
    count_dict = defaultdict(float)
    for pred in pred_list:
        for k, v in pred.items():
            count_dict[k] += 1
    logger.debug(f'sum_vote: {dict(count_dict)}')
    return count_dict


def ensemble(buff: EnsembleBuffer, win: int, size: int, conf_thresh: dict, ensemble_conf_thresh: dict):
    now = int(time.time())
    oldest = now - win
    data = buff.get_range(oldest, now)

    predictions = [(json.loads(x['prediction']), x['model_name']) for x in data]
    predictions = [filter_by_confidence(x, conf_thresh[model_name]) for (x, model_name) in predictions]

    # if len(predictions) < size:
    #     # raise ValueError(f'Not enough data: len({[x["timestamp"] for x in data]}) < {size}')
    #     raise ValueError(f'Not enough data: {len(predictions)=} < {size}')

    # pred = soft_vote(predictions, size)
    pred = hard_vote(predictions)
    pred = filter_by_confidence(pred, ensemble_conf_thresh)

    if len(pred) == 0:
        raise ValueError(f'Not enough data: {len(predictions)=} < {size}')
    meta_data = [json.loads(x['metadata']) for x in data]

    infer_time_ms = [x['inference_time_ms'] for x in meta_data]
    infer_time_ms = sum(infer_time_ms) / len(infer_time_ms)
    inputs = defaultdict(dict)
    for d in meta_data:
        for k, v in d['inputs'].items():
            inputs[k].update(v)
    meta = {
        'timestamp': now,
        'inference_time_ms': infer_time_ms,
        'inputs': dict(inputs),
        'ensemble_size': len(data),
    }
    # logger.debug(f'DEV_DEBUG: {meta}')
    return pred, meta


class Controller(Service):
    def __init__(
        self,
        n_services: int,
        heartbeat_timeout: int,
        execution_plan: str,
        feat_failover: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enable_failover = feat_failover
        self.system_states = SystemStates(n_init_services=n_services)
        self.state_db = StateDb()

        self.n_services = n_services
        self.system_initialized = False
        self.control_message_cool_down = {}

        self.add_sub('heartbeat')

        self.heartbeat_timeout = heartbeat_timeout

        self.counter = 0
        self.failover_cool_down = {}
        self.load_execution_plan(execution_plan)

        self.last_failure_timestamp = 0
        self.when_to_recover_failed_sensor_services = {}

        self._heartbeat_counter = 0

        # ensemble buffer for classification results
        db_file = self.ns_topic_str(self.proc_name, 'ensemble.db')
        self.ensemble_buff = EnsembleBuffer(db_file.replace('/', '_'))
        self.last_ensemble = None

        # UI related
        self.time_series_buffer = TemporalBuffer(size=10)
        self.ui_clients = set()
        self.ui_clients_lock = threading.Lock()

        self.ui_active_subs = Counter()
        self.ui_active_subs_lock = threading.Lock()
        self.infer_service_ctl_topics = []

    def load_execution_plan(self, file):
        with open(file, 'rb') as f:
            plan = toml.load(f)
        for k in ['ok_states', 'service_failures', 'node_failures']:
            plan['failover'][k] = [tuple(sorted(x)) for x in plan['failover'][k]]
        self._service_states.update(plan)

    def _get_service_type_and_short_name(self, state: SystemStateRecord):
        service_name = state.service_name
        service_type, short_name = None, None
        for t, pat in self.service_states['service_types'].items():
            m = re.match(pat, service_name)
            if m is not None:
                service_type, short_name = t, m.group(1)
                break
        return service_type, short_name, state

    def _prune_cool_down(self, oldest_timestamp):
        for k in list(self.failover_cool_down.keys()):
            if self.failover_cool_down[k] < oldest_timestamp:
                del self.failover_cool_down[k]

    def set_service_status(self, node_name: str, service_name: str, deactivated: bool, heartbeat: bool | None = None):
        payload = {'deactivated': deactivated}
        if heartbeat is not None:
            payload['enable_heartbeat'] = heartbeat
        msg = self.make_msg('set', payload, meta={'timestamp': datetime.now().timestamp()})
        topic = f'{node_name}/{service_name}/ctl'
        self.send(topic, msg)

    def fail_sensor(self):
        sensor_services = [
            (node_name, service_name)
            for node_name, service_name in self.system_states.states.keys()
            if ('geo' in service_name or 'mic' in service_name)
            and ('_geo' not in service_name and '_mic' not in service_name)
        ]
        target_node, target_service = random.choice(sensor_services)
        self.set_service_status(target_node, target_service, deactivated=True, heartbeat=False)
        now = datetime.now().timestamp()
        self.last_failure_timestamp = now
        failure_duration = random.randint(30, 120)
        log_msg = {
            'failure': 'sensor',
            'target_node': target_node,
            'target_service': target_service,
            'timestamp': now,
            'failure_length': failure_duration,
        }

        self.when_to_recover_failed_sensor_services[(target_node, target_service)] = now + failure_duration
        logger.debug(f'{log_msg}')

    def recover_failed_sensor(self):
        now = datetime.now().timestamp()
        for k in list(self.when_to_recover_failed_sensor_services.keys()):
            if self.when_to_recover_failed_sensor_services[k] <= now:
                target_node, target_service = k
                self.set_service_status(target_node, target_service, deactivated=False, heartbeat=True)
                log_msg = {
                    'failure': 'sensor_recover',
                    'target_node': target_node,
                    'target_service': target_service,
                    'timestamp': now,
                }
                logger.debug(f'{log_msg}')
                del self.when_to_recover_failed_sensor_services[k]

    def check_and_failover(
        self, node_name: str, services: list[SystemStateRecord], backup_services: list[SystemStateRecord]
    ):
        # services=[
        #    SystemStateRecord(node_name='rs8', service_name='vfm', timestamp=1712627932.567711, deactivated=False),
        #    SystemStateRecord(node_name='rs8', service_name='geo', timestamp=1712627934.875655, deactivated=False),
        #    SystemStateRecord(node_name='rs8', service_name='mic', timestamp=1712627935.023048, deactivated=False)
        # ],
        # backup_services=[
        #    SystemStateRecord(node_name='rs10', service_name='backup/rs8/vfm', timestamp=1712627935.082658, deactivated=True)
        # ],

        # [('infer', 'vfm', SystemStateRecord(node_name='rs8', service_name='vfm', timestamp=1712627932.567711, deactivated=False)), ...]
        all_services_type_shortname = [self._get_service_type_and_short_name(x) for x in services + backup_services]

        # short names of current services on the node
        current_state = tuple(
            sorted(n for _, n, s in all_services_type_shortname if not s.deactivated and n is not None)
        )

        self._prune_cool_down(datetime.now().timestamp() - (self.heartbeat_timeout + 3))

        if current_state in self.service_states['failover']['ok_states']:
            # logger.debug(f'Ok: {node_name}, {current_state}')
            log_msg = {
                'current_state': current_state,
                'node_name': node_name,
                'status': 'ok',
                'timestamp': datetime.now().timestamp(),
            }
            logger.debug(f'{log_msg}')
            return

        cd_key = tuple([node_name, *current_state])
        # ongoing failover; skip until previous failover attempt timeout
        if cd_key in self.failover_cool_down:
            return

        # new service failure
        elif current_state in self.service_states['failover']['service_failures']:
            log_msg = {
                'current_state': current_state,
                'node_name': node_name,
                'status': 'service_failure',
                'timestamp': datetime.now().timestamp(),
            }
            logger.debug(f'{log_msg}')

            logger.debug(f'Service failure: {node_name}, {current_state}. Initiating failover...')
            # put the failure in cool down to avoid the controller being too impatient
            self.failover_cool_down[cd_key] = datetime.now().timestamp()

            # short names of current services except the inconsistent service on the node (e.g., sensor service only)
            current_state_without_failure = [
                n
                for t, n, s in all_services_type_shortname
                if not s.deactivated and n is not None and t != self.service_states['failover']['service_failover']
            ]

            # try to find a standby for failover
            standby = None
            for service_type, short_name, service in all_services_type_shortname:
                if (
                    service_type == self.service_states['failover']['service_failover'] and service.deactivated
                ) and short_name is not None:
                    new_state = tuple(sorted([short_name, *current_state_without_failure]))
                    if new_state in self.service_states['failover']['ok_states']:
                        logger.debug(f'found a standby: activating {service.node_name}/{service.service_name}')
                        # self.deactivate(service.node_name, service.service_name, False)
                        standby = service
                        break
                    else:
                        logger.debug(f'bad standby: {service.node_name}/{service.service_name}')

            if standby is None:
                logger.debug('failed to find a standby')
                return

            # activate standby and disable others
            for service_type, _, service in all_services_type_shortname:
                if service_type != self.service_states['failover']['service_failover']:
                    continue
                elif service != standby and not service.deactivated:
                    self.set_service_status(service.node_name, service.service_name, deactivated=True)
                elif service == standby:
                    self.set_service_status(service.node_name, service.service_name, deactivated=False)
                    log_msg = {
                        'failover': 'backup_activated',
                        'target_node': service.node_name,
                        'target_service': service.service_name,
                        'timestamp': datetime.now().timestamp(),
                    }
                    logger.debug(f'{log_msg}')

        elif current_state in self.service_states['failover']['node_failures']:
            log_msg = {
                'current_state': current_state,
                'node_name': node_name,
                'status': 'node_failure',
                'timestamp': datetime.now().timestamp(),
            }
            logger.debug(f'{log_msg}')

            logger.debug(f'Node failure: {node_name}, {current_state}')
            # put the failure in cool down to avoid the controller being too impatient
            self.failover_cool_down[cd_key] = datetime.now().timestamp()

            # disable services on current node
            for _, _, service in all_services_type_shortname:
                if not service.deactivated:
                    self.set_service_status(service.node_name, service.service_name, deactivated=True, heartbeat=False)

            # try to activate a backup node
            for backup_node in self.system_states.list_backup_nodes():
                activated = False
                for service in self.system_states.list_node_services(backup_node):
                    self.set_service_status(service.node_name, service.service_name, deactivated=False)
                    activated = True
                if activated:
                    logger.debug(f'activated backup node: {backup_node}')
                    # allow new node to stabilize, put all service failures in cool down
                    now = datetime.now().timestamp()
                    for cs in (
                        self.service_states['failover']['service_failures']
                        + self.service_states['failover']['node_failures']
                    ):
                        cs = tuple(sorted(cs))
                        cs = tuple([backup_node, *cs])
                        self.failover_cool_down[cs] = now
                    break
            else:
                logger.debug('failed to find backup node')
        else:
            ss = [s for _, _, s in all_services_type_shortname]
            logger.debug(f'uncaught failure: {node_name}, {current_state}, {ss}')
            # try to reset the node to standby
            # for _, _, service in all_services_type_shortname:
            #     if not service.deactivated:
            #         self.set_service_status(service.node_name, service.service_name, deactivated=False)

        return

    def monitor2(self):
        # get latest liveness state for each (node, service) pair
        now_ns = time.time_ns()
        start_ns = now_ns - int(self.heartbeat_timeout * 1e9)
        states_liveness = self.state_db.get_latest('heartbeat', start_ns, now_ns)
        ignore = ['aciesd', 'noise-detector']
        states_liveness = [ServiceState(**s) for s in states_liveness if s['service'] not in ignore]

        if len(states_liveness) >= self.n_services:
            self.system_initialized = True

        if not self.system_initialized:
            logger.debug(f'>>>> system uninitialized {len(states_liveness)} < {self.n_services}')
            return

        states_noiseness = self.state_db.get_latest('noise', start_ns, now_ns)
        states_noiseness = [ServiceState(**s) for s in states_noiseness]

        node_states_liveness = group_states_by_node(states_liveness)
        node_states_noiseness = group_states_by_node(states_noiseness)

        deps = self.service_states.get('dependencies', {})
        noise_thresh = self.service_states.get('noise_thresh', {})

        for node, ns_liveness in node_states_liveness.items():
            ns_noise = node_states_noiseness.get(node, [])

            if len(ns_noise) > 0:
                ns_liveness = noise_check_node(ns_liveness, ns_noise, noise_thresh)

            msgs = failover_check_node(self.ctrl_topic, node, ns_liveness, deps)
            for msg in msgs:
                topic, payload = msg['topic'], msg['payload']
                cool_down_key = (topic, tuple(payload.items()))
                last_timestamp = self.control_message_cool_down.get(cool_down_key, 0)
                # cool down for same command: 5s
                if now_ns - last_timestamp > int(5 * 1e9):
                    acies_msg = self.make_msg('set', payload)
                    self.send(topic, acies_msg)
                    self.control_message_cool_down[cool_down_key] = now_ns
                    logger.debug(f'>>>>> {msg=}')
                else:
                    logger.debug('>>>>> cooling down; skip')

    def monitor(self):
        if not self.system_states.initialized:
            return

        for node_name in self.system_states.list_live_nodes():
            services = self.system_states.list_node_services(node_name)
            backup_services = self.system_states.list_backup_services(node_name)
            self.check_and_failover(node_name, services, backup_services)

    def handle_messages(self):
        try:
            while True:
                topic, msg = self.msg_q.get_nowait()
                if topic == 'heartbeat' or topic.endswith('noise'):
                    if topic.endswith('noise'):
                        topic = topic.split('/')[-1]

                    # new states
                    node_name = parse_node_name(msg.reply_to)
                    service_name = parse_service_name(msg.reply_to)
                    timestamp_ns = msg.timestamp
                    if topic == 'heartbeat':
                        state = msg.get_metadata()
                    else:
                        state = msg.get_payload()

                    self.state_db.insert_state(topic, node_name, service_name, timestamp_ns, state)

                    node_name, service_name, timestamp, deactivated = get_node_service_timestamp(msg)
                    now = datetime.now().timestamp()
                    if now - timestamp > self.heartbeat_timeout:
                        logger.debug(f'drop stalled heartbeat: {now} - {timestamp} = {now-timestamp}')
                        continue
                    self.system_states.add_record(node_name, service_name, timestamp, msg.get_payload(), deactivated)
                    self._heartbeat_counter += 1
                elif topic.endswith('vehicle'):
                    node_name, model_name = parse_node_model(msg.reply_to)
                    timestamp = int(msg.timestamp / 1e9)
                    pred = msg.get_payload()
                    meta_data = msg.get_metadata()
                    # {'inference_time_ms': 641.014552, 'inputs': {'dvpg-gq-1-shake/geo': {'1722876747': {'channel': 'EH3', 'energy': 113.58657975306764, 'sampling_rate': 200}, '1722876748': {'channel': 'EH3', 'energy': 129.1337135491735, 'sampling_rate': 200}}, 'dvpg-gq-1-shake/mic': {'1722876747': {'channel': 0, 'energy': 137.67991974071091, 'sampling_rate': 16000}, '1722876748': {'channel': 0, 'energy': 132.1544961551719, 'sampling_rate': 16000}}}}
                    energy_meta = self.combine_meta(meta_data['inputs'])
                    geo_thresh = self.service_states.get('energy_threshold', {}).get(node_name, {}).get('geo', 0)
                    mic_thresh = self.service_states.get('energy_threshold', {}).get(node_name, {}).get('mic', 0)

                    e_geo = energy_meta['mean_geo_energy']
                    e_mic = energy_meta['mean_mic_energy']

                    if e_geo >= geo_thresh and e_mic >= mic_thresh:
                        self.ensemble_buff.add_entry(node_name, model_name, timestamp, pred, meta_data)
                    elif np.isnan(e_geo).item() and e_mic >= mic_thresh:
                        self.ensemble_buff.add_entry(node_name, model_name, timestamp, pred, meta_data)
                    elif e_geo >= geo_thresh and np.isnan(e_mic).item():
                        self.ensemble_buff.add_entry(node_name, model_name, timestamp, pred, meta_data)
                elif msg.kind == 'reply':
                    logger.info(f'received reply from {msg.reply_to}, {msg.kind}: {msg.get_payload()}')
                elif msg.kind.startswith('array'):
                    # logger.debug(f'DEV_DEBUG: received timeseires data from {topic}')
                    timestamp = int(msg.timestamp / 1e9)
                    val = msg.get_payload()
                    self.time_series_buffer.add(topic, timestamp, val)
                else:
                    logger.warn(f'unhandled message at {topic}: {msg}')
        except queue.Empty:
            return

    def combine_meta(self, meta_data: dict[str, dict[int, dict]]):
        result = {'label': None, 'distance': None, 'mean_geo_energy': [], 'mean_mic_energy': []}
        oldest_key = 1e15
        for topic, topic_data in meta_data.items():
            try:
                topic = topic.split('/')[-1]
            except IndexError:
                logger.error(f'invalid topic: {topic}')
                logger.error(f'input: {meta_data}')
                raise IndexError
            for k, t_meta in topic_data.items():
                result['label'] = t_meta.get('label', self.service_states.get('ground_truth'))
                result['distance'] = t_meta.get('distance')
                result[f'mean_{topic}_energy'].append(t_meta.get('energy'))
                if float(k) < oldest_key:
                    oldest_key = float(k)
        result['mean_geo_energy'] = np.mean(result['mean_geo_energy'])
        result['mean_mic_energy'] = np.mean(result['mean_mic_energy'])
        result['oldest_timestamp'] = oldest_key
        return result

    def _log_inference_result(self, pred, one_meta, now, ensemble_size=None):
        console_msg = f'E(geo)={one_meta["mean_geo_energy"]:<8.2f} E(mic)={one_meta["mean_mic_energy"]:<8.2f} '
        # FIXME: wrong latency calculate
        latency = now - one_meta['oldest_timestamp']
        console_msg += f'Lat={latency:<4.2f}s'

        if ensemble_size is not None:
            console_msg += f'  Ensemble={ensemble_size}'

        if one_meta['distance'] is not None:
            console_msg += f'  D={one_meta["distance"]:<6.2f}m'

        # predicted label, confidence and ground truth label if available
        if one_meta['label'] is not None:
            console_msg += f' True={one_meta["label"]}'
        else:
            console_msg += ' True=n/a'

        console_msg += ' Detected: '
        for k, v in pred.items():
            console_msg += f'{k} ({v:.2f})  '

        logger.info(console_msg)

    def ensemble(self):
        ensemble_win = self.service_states['ensemble']['win']
        ensemble_size = self.service_states['ensemble']['size']

        if ensemble_size is None or ensemble_win is None:
            logger.warn('ensemble size or window size not set')
            return

        try:
            # ensemble_result, ensemble_meta = self.ensemble_buff.ensemble(ensemble_win, ensemble_size)
            ensemble_result, ensemble_meta = ensemble(
                self.ensemble_buff,
                ensemble_win,
                ensemble_size,
                self.service_states['confidence_threshold'],
                self.service_states['ensemble_confidence_threshold'],
            )
            logger.debug(f'ensemble before conf filtering: {ensemble_result}')
            # pred, confidence = logits_to_predictions(ensemble_result)
            # ensemble_result = filter_by_confidence(ensemble_result, self.service_states['confidence_threshold'])
            logger.debug(f'ensemble after conf filtering: {ensemble_result}')

            if len(ensemble_result) == 0:
                raise ValueError()

            # publish ensemble classification result
            ensemble_msg = self.make_msg('json', ensemble_result, ensemble_meta)

            # send to UI
            self.last_ensemble = ensemble_result
            log_msg = pretty(ensemble_msg.to_dict(), max_seq_length=6, max_width=500, newline='')
            logger.debug(f'ensemble result: {log_msg}')
            one_meta = self.combine_meta(ensemble_meta['inputs'])
            now = datetime.now().timestamp()
            self._log_inference_result(ensemble_result, one_meta, now, ensemble_meta['ensemble_size'])
        except ValueError:
            # not enough data
            logger.debug(f'ensemble buffer: {list(self.ensemble_buff.count())}')
            # clear ensemble
            self.last_ensemble = None
            return

    def log_states_old(self):
        status = 'running' if self.system_states.initialized else 'initializing'
        now = datetime.now().timestamp()
        logger.info(
            f'system states {now:.2f}: {status} '
            f'[ n = {len(self.system_states.states)} / {self.system_states.n_init_services} ] '
            f'{self._heartbeat_counter} heartbeats'
        )
        for k in sorted(self.system_states.states.keys()):
            v = self.system_states.states[k]
            status = 'standby' if v.deactivated else 'running'
            logger.info(f'  {v.node_name:>4}/{v.service_name:25} {status} {v.timestamp:.2f} [ {now-v.timestamp: .2f} ]')

        # log_msg = self.system_states.states.values()
        # logger.debug(f'{log_msg}')

    def log_states(self):
        status = 'running' if self.system_states.initialized else 'initializing'

        now_ns = time.time_ns()
        start_ns = now_ns - int(self.heartbeat_timeout * 1e9)

        states = self.state_db.get_latest('heartbeat', start_ns, now_ns)
        ignore = []
        states = [ServiceState(**s) for s in states if s['service'] not in ignore]
        node_states = group_states_by_node(states)

        logger.info(
            f'system states {(now_ns/1e9):.2f}: {status} '
            # f'[ n = {len(self.system_states.states)} / {self.system_states.n_init_services} ] '
            f'[ n = {sum(len(x) for x in node_states.values())} ] '
            f'{self._heartbeat_counter} heartbeats'
        )

        for k in sorted(node_states.keys()):
            for v in sorted(node_states[k], key=lambda x: x.service):
                status = 'running' if v.is_running() else 'standby'
                logger.info(
                    f'  {v.node:>4}/{v.service:25} {status} {v.timestamp_ns/1e9:.2f} [ {(now_ns-v.timestamp_ns)/1e9:.2f} ]'
                )

        # log_msg = self.system_states.states.values()
        # logger.debug(f'{log_msg}')

    def inject_random_failure(self):
        # introduce failures
        now = datetime.now().timestamp()
        if (
            # how many ongoing failed sensors in the system
            len(self.when_to_recover_failed_sensor_services) < 1
            # probability of failure
            and random.uniform(0, 1) < 0.1
            # interval between failures
            and (now - self.last_failure_timestamp) > 120
        ):
            self.fail_sensor()

        self.recover_failed_sensor()

    def inject_sensor_failure(self, targets):
        for node, mod in targets:
            self.set_service_status(node, mod, deactivated=True, heartbeat=False)
            msg = {'failure': 'injected', 'node': node, 'service': mod, 'timestamp': datetime.now().timestamp()}
            logger.debug(f'{msg}')

    def prune(self):
        now = datetime.now().timestamp()
        self.system_states.prune_outdated(now - self.heartbeat_timeout)

    def send_state_to_ui(self):
        response_json = []

        deps = self.service_states['dependencies']

        now_ns = time.time_ns()
        start_ns = now_ns - int(self.heartbeat_timeout * 1e9)
        states = self.state_db.get_latest('heartbeat', start_ns, now_ns)
        ignore = ['aciesd', 'noise-detector']
        states = [ServiceState(**s) for s in states if s['service'] not in ignore]

        self.infer_service_ctl_topics = [
            make_ctl_topic(x.node, x.service) for x in states if parse_service_name(x.service) in deps
        ]

        # can't use `group_states_by_node` here
        # because here we need put services running on the same node in the same group
        node_states = defaultdict(list)
        for s in states:
            node_states[s.node].append(s)

        # for node_name in self.system_states.list_live_nodes():
        #     services = self.system_states.list_node_services(node_name)
        #     backup_services = self.system_states.list_backup_services(node_name)
        #     current_state = [
        #         {
        #             'host_node': x.node_name,
        #             'service_name': x.service_name,
        #             'deactivated': x.deactivated,
        #         }
        #         for x in (services + backup_services)
        #         # if not x.deactivated
        #     ]
        for node_name, node_services in node_states.items():
            current_state = [
                {'host_node': x.node, 'service_name': x.service, 'deactivated': not x.is_running()}
                for x in node_services
            ]

            for s in current_state:
                try:
                    timestamp, pred = self.ensemble_buff.get_latest(s['host_node'], s['service_name'])
                    # TODO: make timeout a parameter
                    if time.time() - timestamp > 5:
                        pred = None
                except KeyError:
                    pred = None
                if pred is not None:
                    filtered = filter_by_confidence(pred, self.service_states['confidence_threshold'])
                    s['prediction'] = filtered
                else:
                    s['prediction'] = None

            response_data = {
                'current_state': current_state,
                'node_name': node_name,
                'status': 'ok',
                'timestamp': datetime.now().timestamp(),
                'gps': self.service_states.get('gps', {}).get(node_name, ()),
            }

            time_series = {}
            geo_topic = f'{node_name}/geo'
            mic_topic = f'{node_name}/mic'
            try:
                geo_data = self.time_series_buffer.pop([geo_topic], 1)
                geo_data = list(geo_data[geo_topic].values())[0]
                geo_data = geo_data[::2]
                time_series['geo'] = geo_data
            except ValueError:
                pass

            try:
                mic_data = self.time_series_buffer.pop([mic_topic], 1)
                mic_data = list(mic_data[mic_topic].values())[0]
                mic_data = mic_data[::160]
                time_series['mic'] = mic_data
            except ValueError:
                pass

            response_data['raw_sensor'] = time_series

            response_json.append(response_data)

        # ensemble result
        response_json.append(
            {
                'current_state': {
                    'host_node': self.namespace,
                    'service_name': 'ensemble',
                    'deactivated': False,
                    'prediction': self.last_ensemble,
                },
                'node_name': self.namespace,
                'status': 'ok',
                'timestamp': datetime.now().timestamp(),
                'raw_sensor': {},
                'gps': None,
            }
        )
        response_json = json.dumps(response_json)
        with self.ui_clients_lock:
            if len(self.ui_clients) == 0:
                return
            for client in self.ui_clients.copy():
                # logger.debug(f'send states to UI {client}: {response_json}')
                try:
                    client.send(response_json)
                except Exception as e:
                    logger.error(f'Error sending to client: {e}')
                    self.ui_clients.remove(client)

    def ui_handler(self, websocket):
        logger.debug('DEV_DEBUG: New client connected')
        try:
            with self.ui_clients_lock:
                self.ui_clients.add(websocket)
                logger.debug(f'ui_clients: {self.ui_clients}')
            for message in websocket:
                # logger.debug(f'>>>>>> {message}')
                # {"action":"subscribe","node_id":"rs10"}
                # {"action":"unsubscribe","node_id":"rs10"}
                # {
                #     'action': 'control',
                #     'messages': [
                #         {
                #             'to': 'rs10/vfm/ctl',
                #             'msg': {
                #                 'kind': 'set',
                #                 'timestamp': 1724530726581000000,
                #                 'reply_to': 'cp/controller/ctl',
                #                 'payload': {'n': 10},
                #                 'metadata': {},
                #             },
                #         },
                #         {
                #             'to': 'rs3/vfm/ctl',
                #             'msg': {
                #                 'kind': 'set',
                #                 'timestamp': 1724530726581000000,
                #                 'reply_to': 'cp/controller/ctl',
                #                 'payload': {'n': 10},
                #                 'metadata': {},
                #             },
                #         },
                #     ],
                # }
                #
                try:
                    data = json.loads(message)
                    # logger.debug(f'DEV_DEBUG: receive message: {data}')
                    action = data.get('action')

                    if action in ['subscribe', 'unsubscribe']:
                        # topic change
                        node_id = data.get('node_id')
                        geo_topic = f'{node_id}/geo'
                        mic_topic = f'{node_id}/mic'
                        if action == 'subscribe':
                            with self.ui_active_subs_lock:
                                self.ui_active_subs[geo_topic] += 1
                                self.ui_active_subs[mic_topic] += 1
                                logger.debug(f'UI Subscribers: (subscribe) {self.ui_active_subs}')
                            self.add_sub(geo_topic)
                            self.add_sub(mic_topic)
                        elif action == 'unsubscribe':
                            with self.ui_active_subs_lock:
                                self.ui_active_subs[geo_topic] -= 1
                                self.ui_active_subs[mic_topic] -= 1
                                logger.debug(f'UI Subscribers: (unsubscribe) {self.ui_active_subs}')
                            # last active sub, drop the subscription
                            if self.ui_active_subs[geo_topic] == 0:
                                self.remove_sub(geo_topic)
                                del self.time_series_buffer._data[geo_topic]
                            if self.ui_active_subs[mic_topic] == 0:
                                self.remove_sub(mic_topic)
                                del self.time_series_buffer._data[mic_topic]
                    elif action == 'control':
                        # forward control messages
                        # FIXME: the `to` field from the UI hard-coded `vfm`.
                        # So ignore the specific messages and only use the msg content for now.
                        # The controller handles getting all the running classifiers and forward the message.
                        msg = data['messages'][0]['msg']
                        logger.debug(f'>>>>> {self.infer_service_ctl_topics=}')
                        for topic_to in self.infer_service_ctl_topics:
                            acies_msg = self.make_msg(msg['kind'], msg['payload'])
                            self.send(topic_to, acies_msg)
                            logger.debug(f'>>>>> {topic_to}: {acies_msg}')
                    else:
                        logger.error(f'DEV_DEBUG: Expect action: subscribe, unsubscribe, control. Got: {action}')
                except json.JSONDecodeError:
                    logger.error(f'DEV_DEBUG: received invalid JSON: {message}')
        finally:
            # FIXME: potential dead lock on exit
            with self.ui_clients_lock:
                self.ui_clients.remove(websocket)
            logger.debug(f'DEV_DEBUG: Client disconnected. Total clients: {len(self.ui_clients)}')

    def ui_main(self):
        port = 9001
        host = '127.0.0.1'
        # host = '0.0.0.0'
        logger.debug(f'connect to UI through websocket: {host}:{port}')
        with ws_server.serve(self.ui_handler, host=host, port=port) as server:
            self.ui_server = server
            server.serve_forever()
        logger.debug('Exited ui_main')

    def shutdown(self):
        self.ui_server.shutdown()

    def run(self):
        logger.info(f'service_states: {self.service_states}')

        # exp 1
        # self.schedule(6 * 60, self.inject_sensor_failure, ([('rs5', 'mic')],))
        # self.schedule(7 * 60, self.inject_sensor_failure, ([('rs7', 'geo')],))
        # self.schedule(8 * 60, self.inject_sensor_failure, ([('rs10', 'mic'),]))
        # self.schedule(9 * 60, self.inject_sensor_failure, ([('rs6', 'geo')],))

        # exp 2: kill nodes
        # self.schedule(3 * 60, self.inject_sensor_failure, ([('rs10', 'mic'), ('rs10', 'geo')],))
        # self.schedule(4 * 60, self.inject_sensor_failure, ([('rs5', 'mic'), ('rs5', 'geo')],))

        self.schedule(0.1, self.handle_messages, periodic=True)

        # self.schedule(2, self.log_states_old, periodic=True)
        self.schedule(2, self.log_states, periodic=True)
        self.schedule(1, self.prune, periodic=True)

        # failover
        if self.enable_failover:
            self.schedule(1, self.monitor, periodic=True)
        self.schedule(1, self.monitor2, periodic=True)

        # ensemble
        self.schedule(2, self.ensemble, periodic=True)

        # UI
        self.ui_thread = threading.Thread(target=self.ui_main)
        self.ui_thread.start()
        self.schedule(1, self.send_state_to_ui, periodic=True)


@click.command()
@common_options
@click.option('--n_services', type=int, default=1, help='Wait until the number of services all have joined the system.')
@click.option('--heartbeat_timeout', type=int, default=5, help='Seconds to wait before invalidate last heartbeat.')
@click.option('--execution-plan', type=click.Path(exists=True), help='Execution plan TOML file.')
@click.option('--feat-failover', is_flag=True, show_default=True, default=False, help='Enable failover functionality.')
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    deactivated,
    n_services,
    heartbeat_timeout,
    execution_plan,
    feat_failover,
):
    init_logger(f'{namespace}_{proc_name}.log', name='acies')
    z_conf = get_zconf(mode, connect, listen)

    controller = Controller(
        n_services,
        heartbeat_timeout,
        execution_plan,
        feat_failover,
        conf=z_conf,
        namespace=namespace,
        proc_name=proc_name,
        connect=connect,
        listen=listen,
        topic=topic,
        deactivated=deactivated,
        enable_heartbeat=False,
    )
    controller.start()
