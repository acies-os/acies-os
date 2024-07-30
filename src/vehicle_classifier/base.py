import logging
import queue
import threading
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import click
import numpy as np
from acies.core import AciesMsg, Service, common_options, get_zconf, init_logger, pretty
from acies.vehicle_classifier.buffer import StreamBuffer, TemporalEnsembleBuff
from acies.vehicle_classifier.utils import TimeProfiler, update_sys_argv

logger = logging.getLogger('acies.infer')

LABEL_TO_STR = {
    0: 'miata',
    1: 'gle350',
    2: 'mustang',
    3: 'cx30',
}


def time_diff_decorator(func):
    last_call = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        current_time = time.time()
        if func.__name__ in last_call:
            time_diff = current_time - last_call[func.__name__]
            logger.debug(f'Time since last call to {func.__name__}: {time_diff:.6f} seconds')
        else:
            logger.debug(f'First call to {func.__name__}')
        last_call[func.__name__] = current_time
        return func(*args, **kwargs)

    return wrapper


def get_twin_topic(topic: str) -> str:
    """Convert between physical and digital twin topic."""
    if topic.startswith('twin/'):
        # twin/ns1/n1/ctrl -> ns1/n1/ctrl
        return topic.removeprefix('twin/')
    else:
        # ns1/n1/ctrl -> twin/ns1/n1/ctrl
        return 'twin/' + topic


class Classifier(Service):
    def __init__(
        self,
        classifier_config_file,
        twin_model,
        twin_buff_len: int,
        sync_interval: int,
        feature_twin: bool,
        *args,
        **kwargs,
    ):
        # pass other args to parent type
        super().__init__(*args, **kwargs)

        # features
        self.feature_twin = feature_twin

        # init digital twin
        if self.feature_twin:
            self.twin_init(twin_model, twin_buff_len)
            self.sync_interval = sync_interval

        # buffer 10s of data for each topic
        self.buffer = StreamBuffer(size=10)

        # classification result ensemble buffer
        self.ensemble_buff = TemporalEnsembleBuff(buff_size=20)

        # how many input messages the model needs to run inference once
        # each message contains 1s of data:
        #     seismic  :    200 samples
        #     acoustic : 16_000 samples
        self.input_len = 2

        # the topic we publish inference results to
        self.pub_topic = self.ns_topic_str('vehicle')
        logger.info(f'classification result published to {self.pub_topic}')

        # your inference model
        classifier_config_file = Path(classifier_config_file)
        assert classifier_config_file.exists(), f'{classifier_config_file} does not exist'

        self.modalities = []
        self.model = self.load_model(classifier_config_file)

    def twin_init(self, twin_model, twin_buff_len):
        self.is_digital_twin = self.ctrl_topic.startswith('twin/')
        if self.is_digital_twin:
            logger.info('running as digital twin')
            self.service_states['enable_heartbeat'] = False
        else:
            logger.info('running as physical twin')

        # digital twin ctrl parameters
        self.service_states['twin/model'] = twin_model
        self.service_states['twin/sync_method'] = 'fixed_interval'
        self.service_states['twin/buff_len'] = twin_buff_len

        # dict that holds the latest msg from each topic, `self.sync_with_twin`
        # will send messages in this dict to the digital twin
        self._sync_latest: dict[str, AciesMsg] = {}
        self._sync_latest_lock = threading.Lock()

    def twin_sync_register(self, topic, msg):
        with self._sync_latest_lock:
            self._sync_latest[topic] = msg

    def load_model(self, *args, **kwargs):
        """Load model from give path."""
        raise NotImplementedError

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
                if int(k) < oldest_key:
                    oldest_key = int(k)
        result['mean_geo_energy'] = np.mean(result['mean_geo_energy'])
        result['mean_mic_energy'] = np.mean(result['mean_mic_energy'])
        result['oldest_timestamp'] = oldest_key
        return result

    def get_keys_per_node(self, modalities):
        keys = list(self.buffer._data.keys())
        nodes = set()
        for k in keys:
            for m in modalities:
                k = k.removesuffix(m).rstrip('/')
            nodes.add(k)
        ns_keys = {n: [f'{n}/{m}' for m in modalities] for n in nodes}
        return ns_keys

    # @time_diff_decorator
    def run_inference(self):
        node_keys = self.get_keys_per_node(self.modalities)
        for node, keys in node_keys.items():
            try:
                samples, meta_data = self.buffer.get(keys, self.input_len)
            except ValueError:
                # not enough data
                logger.debug(f'not enough data for {node}')
                return

            # run inference and record execution time
            with TimeProfiler() as timer:
                result = self.infer(samples)
            infer_time_ms = timer.elapsed_time_ns / 1e6

            # log inference result
            result = {LABEL_TO_STR[k]: v.item() for k, v in result.items()}
            metadata = {'inference_time_ms': infer_time_ms, 'inputs': dict(meta_data)}
            msg = self.make_msg('json', result, metadata)
            log_msg = pretty(msg.to_dict(), max_seq_length=6, max_width=500, newline='')
            logger.debug(f'inference result: {log_msg}')

            # log predicted label and confidence
            pred, confidence = max(result.items(), key=lambda x: x[1])
            one_meta = self.combine_meta(meta_data)
            log_msg = {
                'pred_label': pred,
                'confidence': confidence,
                'true_label': one_meta['label'],
                'distance': one_meta['distance'],
                'energy_geo': one_meta['mean_geo_energy'],
                'energy_mic': one_meta['mean_mic_energy'],
            }
            logger.info(f'{log_msg}')

            # perform temporal ensemble
            if self.feature_twin:
                self.twin_temp_ensemble(node, msg)
            else:
                self.send(self.pub_topic, msg)

    def twin_temp_ensemble(self, node, msg):
        self.ensemble_buff.add(msg)
        try:
            buff_len = int(self.service_states['twin/buff_len'])
            min_input_t = min([min(int(vv) for vv in v.keys()) for v in msg.get_metadata()['inputs'].values()])
            ensemble_result, ensemble_meta = self.ensemble_buff.ensemble(
                min_input_t,
                # give it an extra second to accommodate the fluctuation
                self.input_len * (buff_len - 1),
                buff_len,
            )
            pred, confidence = max(ensemble_result.items(), key=lambda x: x[1])
            if self.is_digital_twin:
                for k, v in self.service_states.items():
                    if k.startswith('twin/'):
                        ensemble_meta[k] = v
            # publish ensemble classification result
            ensemble_msg = self.make_msg('json', ensemble_result, meta=ensemble_meta)
            self.send(f'{node}/vehicle', ensemble_msg)
            pretty(ensemble_msg.to_dict(), max_seq_length=6, max_width=500, newline='')
            # logger.debug(f'ensemble result: {log_msg}')
            one_meta = self.combine_meta(ensemble_meta['inputs'])
            # use current message timestamp as now
            now = msg.timestamp
            self._log_inference_result(pred, confidence, one_meta, now, ensemble_meta['ensemble_size'])
        except ValueError:
            # not enough data
            logger.debug(f'temporal ensemble buffer: {list(self.ensemble_buff._data.keys())}')
            return

    def _log_inference_result(self, pred, confidence, one_meta, now, ensemble_size=None):
        latency = now - one_meta['oldest_timestamp']
        console_msg = f'detected {pred:<7} ({confidence:.4f}): '

        # predicted label, confidence and ground truth label if available
        if one_meta['label'] is not None:
            console_msg += f'truth={one_meta["label"]:<7} '
        else:
            console_msg += f'truth={"n/a":<7} '

        if one_meta['distance'] is not None:
            console_msg += f'D={one_meta["distance"]:<6.2f}m '
        # else:
        #     console_msg += f'D={"n/a":<6}m, '
        console_msg += f'E(geo)={one_meta["mean_geo_energy"]:<8.2f} E(mic)={one_meta["mean_mic_energy"]:<8.2f} '
        console_msg += f'L={latency:<4.2f}s'
        if ensemble_size is not None:
            console_msg += f' Ensemble={ensemble_size}'
        logger.info(console_msg)

    def infer(self, samples):
        raise NotImplementedError()

    @staticmethod
    def concat(arrays: dict[int, np.ndarray]):
        # the samples in v are sorted by timestamp
        assert list(arrays.keys()) == sorted(arrays.keys())
        return np.concatenate(list(arrays.values()))

    def twin_sync(self):
        if self.is_digital_twin:
            return

        to_sync: dict[str, AciesMsg] = {}
        with self._sync_latest_lock:
            keys_to_sync = list(self._sync_latest.keys())
            for k in keys_to_sync:
                to_sync[k] = self._sync_latest.pop(k)

        for topic, _msg in to_sync.items():
            # deep copy the msg to avoid modification
            msg = AciesMsg.from_bytes(_msg.to_bytes())

            # sync_topic = 'cp/dtwin_ctrl/ctrl'
            sync_topic = get_twin_topic(topic)
            # add twin sync meta data including the sync method, timestamp, and msg_id
            metadata = msg.get_metadata()
            metadata['twin/sync_method'] = self.service_states['twin/sync_method']
            metadata['twin/sync_timestamp'] = datetime.now().timestamp()
            msg.set_metadata(metadata)
            # msg.metadata['twin/sync_msg_id'] = self.new_msg_id()
            self.send(sync_topic, msg)
            logger.debug(f'synced msg to {sync_topic}: {msg.timestamp}')

    # @time_diff_decorator
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
            now = int(datetime.now().timestamp())
            # logger.debug(f'handle_message: {timestamp=}, lat={now-timestamp}, qsize={self.msg_q.qsize()}')
            array = np.array(msg.get_payload())
            mod = 'geo' if topic.endswith('geo') else 'mic'

            # filter out low energy messages
            energy = np.std(array)
            thresh = self.service_states.get(f'{mod}_energy_thresh', 0.0)
            if energy < thresh:
                logger.debug(f'energy below threshold: {energy} < {thresh} at {topic}; drop message: {msg}')
                return
            metadata = msg.get_metadata()
            metadata['energy'] = energy

            self.buffer.add(topic, timestamp, array, metadata)
            if self.feature_twin:
                # stage the latest msg for each topic to sync with the twin
                self.twin_sync_register(topic, msg)
        else:
            logger.info(f'unhandled msg received at topic {topic}: {msg}')

    def log_activate_status(self):
        if self.service_states.get('deactivated', False):
            logger.debug('currently deactivated, standing by')

    def run(self):
        self.schedule(2, self.log_activate_status, periodic=True)
        self.schedule(0.1, self.handle_message, periodic=True)
        self.schedule(1, self.run_inference, periodic=True)
        if self.feature_twin:
            self.schedule(self.sync_interval, self.twin_sync, periodic=True)
        self._scheduler.run()


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=str)
@click.option('--sync-interval', help='Sync interval in seconds', type=int, default=1)
@click.option('--feature-twin', help='Enable digital twin features', is_flag=True, default=False)
@click.option('--twin-model', help='Model used in the digital twin', type=str, default='multimodal')
@click.option('--twin-buff-len', help='Buffer length in the digital twin', type=int, default=2)
@click.option('--heartbeat-interval-s', help='Heartbeat interval in seconds', type=int, default=5)
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
def main(
    mode,
    connect,
    listen,
    topic,
    namespace,
    proc_name,
    weight,
    sync_interval,
    model_args,
    feature_twin,
    twin_model,
    twin_buff_len,
    heartbeat_interval_s,
):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    init_logger(f'{namespace}_{proc_name}.log', name='acies.infer')
    z_conf = get_zconf(mode, connect, listen)

    # initialize the class
    clf = Classifier(
        classifier_config_file=weight,
        sync_interval=sync_interval,
        twin_model=twin_model,
        twin_buff_len=twin_buff_len,
        conf=z_conf,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        feature_twin=feature_twin,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # start
    clf.start()
