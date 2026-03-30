import json
import logging
import queue
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from acies.buffers import EnsembleBuffer
from acies.core import AciesMsg, Service, common_options, get_zconf, init_logger

logger = logging.getLogger('acies.noise_detector')


def get_node_name(msg: AciesMsg) -> str:
    reply_to = msg.reply_to
    return reply_to.split('/')[0]


class NoiseDetector(Service):
    def __init__(self, win_size: int, *args, **kwargs):
        # pass other args to parent type
        logger.debug(f'Initializing NoiseDetector')
        super().__init__(*args, **kwargs)

        self.buffer = EnsembleBuffer(Path.home() / '.acies' / 'noise.db')
        self.win_size = win_size

        # the topic we publish inference results to
        self.pub_topic = self.ns_topic_str('noise')
        logger.info(f'noise info published to {self.pub_topic}')

    def handle_message(self):
        try:
            topic, msg = self.msg_q.get_nowait()
            assert isinstance(msg, AciesMsg)
        except queue.Empty:
            return

        # if deactivated, drop the message
        if self.service_states.get('deactivated', False):
            return

        if any(topic.endswith(x) for x in ['geo', 'mic']):
            # msg.timestamp is in ns
            timestamp = int(msg.timestamp / 1e9)
            array = np.array(msg.get_payload())
            mod = 'geo' if topic.endswith('geo') else 'mic'
            energy = np.std(array).item()
            node_name = get_node_name(msg)
            self.buffer.add_entry(node_name, 'noise_detector', timestamp, {mod: energy}, None)
        else:
            logger.info(f'unhandled msg received at topic {topic}: {msg}')

    def detect(self):
        now = int(datetime.now().timestamp())
        oldest = now - self.win_size
        # samples with timestamp \in [oldest, now]
        samples = self.buffer.get_range(oldest, now)

        mod_energy = {
            'geo': [],
            'mic': [],
        }

        # parse each sample prediction
        for sample in samples:
            sample_energy = json.loads(sample['prediction'])  # json str -> dict
            mod = next(iter(sample_energy))  # each sample is geo xor mic
            mod_energy[mod].append(sample_energy[mod])

        # compute the average energy for each modality
        average_energy = {
            'timestamp': now,
            'geo': np.mean(mod_energy['geo']) if len(mod_energy['geo']) > 0 else 0,
            'mic': np.mean(mod_energy['mic']) if len(mod_energy['mic']) > 0 else 0,
        }

        logger.debug(f"{now}, {average_energy['geo']}, {average_energy['mic']}")

        msg = self.make_msg('json', average_energy)
        self.send(self.pub_topic, msg)

    def run(self):
        self.schedule(0.1, self.handle_message, periodic=True)
        self.schedule(1, self.detect, periodic=True)


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--win-size', help='Window size in seconds', type=int)
@click.option('--heartbeat-interval-s', help='Heartbeat interval in seconds', type=int, default=5)
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    win_size,
    heartbeat_interval_s,
    deactivated=False,  # !TODO unsure why deactivated is needed
):
    init_logger(f'{namespace}_{proc_name}.log', name='acies.noise_detector')
    z_conf = get_zconf(mode, connect, listen)

    # initialize the class
    clf = NoiseDetector(
        win_size=win_size,
        conf=z_conf,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # start
    clf.start()
