import logging
import queue
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import click
from acies.node.logging import init_logger
from acies.node.net import common_options, get_zconf
from acies.node.service import AciesMsg, Service

logger = logging.getLogger('acies.twin')


def get_twin_topic(topic: str) -> str:
    """Convert between physical and digital twin topic."""
    if topic.startswith('twin/'):
        # twin/ns1/n1/ctrl -> ns1/n1/ctrl
        return topic.removeprefix('twin/')
    else:
        # ns1/n1/ctrl -> twin/ns1/n1/ctrl
        return 'twin/' + topic


@dataclass
class TwinResultBuffer:
    buff_size: int = 10
    # topic -> timestamp -> pred
    _data: dict[str, dict[int, dict]] = field(default_factory=lambda: defaultdict(dict))
    _meta_data: dict[str, dict[int, dict]] = field(default_factory=lambda: defaultdict(dict))

    # topic -> timestamp -> config
    _config: dict[str, dict] = field(default_factory=dict)

    # digital twin topic -> physical twin topic
    _mapping: dict[str, str] = field(default_factory=dict)

    def add(self, msg: AciesMsg):
        digital_ctl_topic = msg.reply_to
        digital_config = {k: v for k, v in msg.meta.items() if k.startswith('twin/')}

        physical_ctl_topic = []
        timestamp = []
        for _, topic_msgs in msg.meta['inputs'].items():
            for t, msg_t in topic_msgs.items():
                timestamp.append(t)
                physical_ctl_topic.append(msg_t['twin/sync_msg_id'])

        timestamp = min(timestamp)
        physical_ctl_topic = physical_ctl_topic[0].split('/msg/')[0] + '/ctrl'

        self._data[digital_ctl_topic][timestamp] = msg.payload
        self._meta_data[digital_ctl_topic][timestamp] = msg.meta
        self._config[digital_ctl_topic] = digital_config
        self._mapping[digital_ctl_topic] = physical_ctl_topic

    def get_preds(self, win_size: int):
        latest_t = {k: max(v.keys()) for k, v in self._data.items()}
        latest_t = max(latest_t.values())

        preds = defaultdict(dict)
        preds_latency = defaultdict(dict)
        for m, m_data in self._data.items():
            for t, v in m_data.items():
                if t < latest_t - win_size:
                    continue
                # convert logits to label
                preds[m][t] = max(v.items(), key=lambda x: x[1])[0]
                t_infer = self._meta_data[m][t]['timestamp']
                t_oldest_inptut = min([min(v.keys()) for v in self._meta_data[m][t]['inputs'].values()])
                preds_latency[m][t] = t_infer - t_oldest_inptut
                # logger.debug(f'-----> {preds_latency[m][t]}')

        mean_latency = {k: sum(v.values()) / len(v) for k, v in preds_latency.items()}
        return dict(preds), mean_latency


class TwinCtl(Service):
    def __init__(self, n_twins, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.twin_result_buff = TwinResultBuffer(buff_size=10)
        self.n_twins = n_twins
        self._last_best_config = {}

    def handle_messages(self):
        # dispatch function
        try:
            while True:
                topic, msg = self.msg_q.get_nowait()
                assert isinstance(msg, AciesMsg)
                if msg.msg_type == 'classification' and msg.reply_to.startswith('twin/'):
                    self.twin_result_buff.add(msg)
                elif msg.msg_type.endswith('param_get/reply'):
                    if msg.reply_to not in self._last_best_config:
                        self._last_best_config[msg.reply_to] = msg.payload
                        logger.debug(f'set initial config from physical twin: {msg.reply_to}')
                else:
                    logger.debug(f'unknown message type: {msg}')
        except queue.Empty:
            # logger.debug('no message to handle')
            return

    def optimize(self):
        if len(self.twin_result_buff._data) < self.n_twins:
            logger.debug(f'waiting results from digital twins: {list(self.twin_result_buff._data.keys())}')
            return

        preds, mean_latency = self.twin_result_buff.get_preds(120)
        # logger.debug(f'select best config: {preds}')
        if 'ground_truth' not in self.service_states:
            acc = {k: Counter(v.values()) for k, v in preds.items()}
            logger.debug(f'ground_truth unavailable, pred_label distribution: {acc}')
        else:
            true_label = self.service_states.get('ground_truth')
            acc = {k: Counter(v.values()) for k, v in preds.items()}
            latest_timestamp = {k: max(v.keys()) for k, v in preds.items()}
            total = {k: sum(v.values()) for k, v in acc.items()}
            acc = {k: v.get(true_label, 0) / sum(v.values()) for k, v in acc.items()}
            acc = dict(sorted(acc.items(), key=lambda x: x[1], reverse=True))

            log_msg = []
            for k, v in acc.items():
                line = {
                    'name': k,
                    'accuracy': v,
                    'latency': mean_latency[k],
                    'N': total[k],
                    'timestamp': latest_timestamp[k],
                }
                for kk, vv in self.twin_result_buff._config[k].items():
                    line[kk] = vv
                log_msg.append(line)
            log_msg = sorted(log_msg, key=lambda x: (x['accuracy'] / x['latency']), reverse=True)
            logger.debug(f'perf eval: {log_msg}')

            best_twin_name = log_msg[0]['name']
            physical_twin_topic = self.twin_result_buff._mapping[best_twin_name]
            best_config = self.twin_result_buff._config[best_twin_name]

            # init _last_best_config with physical_twin's current config
            if physical_twin_topic not in self._last_best_config:
                msg = self.make_msg('param_get', best_config, {'timestamp': datetime.now().timestamp()})
                self.send(physical_twin_topic, msg)
                logger.debug(f'getting initial config from physical twin: {physical_twin_topic}')
                return

            if best_config == self._last_best_config.get(physical_twin_topic):
                line_config_change = f'no change to the best config: {best_config}'
                logger.debug(line_config_change)
            else:
                msg = self.make_msg('param_set', best_config, {'timestamp': datetime.now().timestamp()})
                self.send(physical_twin_topic, msg)
                line_config_change = f'reconfigure {physical_twin_topic} from {self._last_best_config.get(physical_twin_topic)} to {best_config}'
                logger.debug(line_config_change)
                self._last_best_config[physical_twin_topic] = best_config

            with open('twin_perf.log', 'w') as f:
                f.write(line_config_change + '\n\n')
                f.write(f'{" ordered by accuracy/latency ":=^50}\n\n')
                for x in log_msg:
                    line = f'{" " + x["name"] + " ":-^50}\n'
                    line += f'{"accuracy":>20} = {x["accuracy"]*100:4.2f} %\n'
                    line += f'{"latency":>20} = {x["latency"]:4.2f} s\n'
                    line += f'{"N":>20} = {x["N"]}\n'
                    line += f'{"timestamp":>20} = {x["timestamp"]}\n'
                    for k in x.keys():
                        if k not in ['name', 'accuracy', 'latency', 'N', 'timestamp']:
                            line += f'{k:>20} = {x[k]}\n'
                    line += '\n'
                    f.write(line)
                f.write(f'{" end ":=^50}\n')

    def run(self):
        self.sched_periodic(0.1, self.handle_messages, ())
        self.sched_periodic(2, self.optimize, ())
        self._scheduler.run()


@click.command()
@common_options
@click.option('--n_twins', type=int, default=2, help='Number of twins to wait for.')
def main(mode, connect, listen, topic, namespace, proc_name, deactivated, n_twins):
    init_logger(f'{namespace}_{proc_name}.log', get_logger='acies')
    z_conf = get_zconf(mode, connect, listen)

    twinctl = TwinCtl(
        n_twins=n_twins,
        conf=z_conf,
        namespace=namespace,
        proc_name=proc_name,
        connect=connect,
        listen=listen,
        topic=topic,
        deactivated=deactivated,
        enable_heartbeat=False,
    )
    twinctl.start()
