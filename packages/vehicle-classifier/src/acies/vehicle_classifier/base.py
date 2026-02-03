import json
import logging
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path

import click
import numpy as np

from acies.buffers import EnsembleBuffer
from acies.core import AciesMsg, Service, common_options, get_zconf, init_logger, pretty
from acies.vehicle_classifier.buffer import StreamBuffer, TemporalEnsembleBuff
from acies.vehicle_classifier.utils import TimeProfiler, update_sys_argv

logger = logging.getLogger('acies.infer')

# ICT experiment
# LABEL_TO_STR = {
#     0: 'miata',
#     1: 'gle350',
#     2: 'mustang',
#     3: 'cx30',
# }

# GQ experiment
LABEL_TO_STR = {
    0: 'polaris',
    1: 'warthog',
    2: 'truck',
    3: 'husky',
}


def soft_vote(pred_list: list[dict[str, float]]) -> dict[str, float]:
    """Perform soft voting on a list of predictions.

    Args:
        pred_list (list[dict[str, float]]): List of prediction dictionaries.

    Returns:
        dict[str, float]: A dictionary with the mean prediction scores.
    """
    if not pred_list:
        return {}
    sum_dict = defaultdict(float)
    # count_dict = defaultdict(int)
    # for pred in pred_list:
    #     for k, v in pred.items():
    #         sum_dict[k] += v
    #         count_dict[k] += 1
    # mean_pred = {k: sum_dict[k] / count_dict[k] for k in sum_dict}
    # logger.debug(f'soft_vote: {dict(count_dict)}')
    for pred in pred_list:
        for k, v in pred.items():
            sum_dict[k] += v
    mean_pred = {k: sum_dict[k] / len(pred_list) for k in sum_dict}
    return mean_pred


def ensemble(buff: EnsembleBuffer, win: int, size: int, conf_thresh: dict):
    """Perform ensemble voting on the buffered predictions.

    Args:
        buff (EnsembleBuffer): A buffer containing the predictions to ensemble.
        win (int): The time window (in seconds) to consider for ensembling.
        size (int): The minimum number of predictions required for ensembling.
        conf_thresh (dict): A dictionary of confidence thresholds for each class.

    Raises:
        ValueError: If there are not enough predictions for ensembling.

    Returns:
        tuple[dict[str, float], dict]: The ensemble prediction and metadata.
    """
    now = int(time.time())
    oldest = now - win
    data = buff.get_range(oldest, now)

    predictions = [json.loads(x['prediction']) for x in data]

    if len(predictions) < size:
        raise ValueError(f'Not enough data: {len(predictions)=} < {size}')

    pred = soft_vote(predictions)
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


def time_diff_decorator(func):
    """Decorator to log the time difference between consecutive calls to a function.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """
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
    """Convert between physical and digital twin topic.

    Args:
        topic (str): The topic to convert.

    Returns:
        str: The converted topic.
    """
    if topic.startswith('twin/'):
        # twin/ns1/n1/ctrl -> ns1/n1/ctrl
        return topic.removeprefix('twin/')
    else:
        # ns1/n1/ctrl -> twin/ns1/n1/ctrl
        return 'twin/' + topic


class Classifier(Service):
    """A template classifier for vehicle data.
    This class should be subclassed and the `load_model` and `infer` methods implemented.
    """

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
        """Initialize the Classifier.

        Args:
            classifier_config_file (str): Path to the classifier configuration file.
            twin_model (str): The digital twin model to use.
            twin_buff_len (int): The buffer length for the digital twin.
            sync_interval (int): The synchronization interval for the digital twin.
            feature_twin (bool): Whether to use the digital twin features.
        """
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

        db_file = self.ns_topic_str(self.proc_name, 'ensemble.db')
        self.ensemble_buff_db = EnsembleBuffer(db_file.replace('/', '_'))

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
        """Combine metadata from two topics.

        Args:
            meta_data (dict[str, dict[int, dict]]): The metadata to combine.

        Raises:
            IndexError: If the topic format is invalid.

        Returns:
            dict: The combined metadata.
        """
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
        """Get the keys for each node based on the modalities.

        Args:
            modalities (list[str]): The list of modalities to consider.

        Returns:
            dict[str, list[str]]: A dictionary mapping each node to its keys.
        """
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
                # self.send(self.pub_topic, msg)
                self.temp_ensmeble(node, msg)

    def temp_ensmeble(self, node, msg):
        """Perform temporal ensemble on the classification results.

        Args:
            node (str): The node name.
            msg (Message): The message containing the classification results.

        Raises:
            ValueError: If the ensemble buffer is not sufficient.
        """
        model_name = self.proc_name
        self.ensemble_buff_db.add_entry(
            node, model_name, int(msg.timestamp / 1e9), msg.get_payload(), msg.get_metadata()
        )
        ensemble_win = int(self.service_states.get('twin/buff_len', 1))
        ensemble_size = int(self.service_states.get('twin/ensemble_size', 1))
        try:
            # ensemble_result, ensemble_meta = self.ensemble_buff.ensemble(ensemble_win, ensemble_size)
            ensemble_result, ensemble_meta = ensemble(self.ensemble_buff_db, ensemble_win, ensemble_size, {})

            if len(ensemble_result) == 0:
                raise ValueError()

            # publish ensemble classification result
            ensemble_msg = self.make_msg('json', ensemble_result, ensemble_meta)
            topic_to = f'{node}/vehicle'
            self.send(topic_to, ensemble_msg)
            logger.debug(f'>>>>> {topic_to} [{ensemble_meta["ensemble_size"]}]: {ensemble_msg}')

        except ValueError:
            # not enough data
            logger.debug(f'ensemble buffer: {list(self.ensemble_buff_db.count())}')
            # clear ensemble
            self.last_ensemble = None
            return

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
        """Log the inference result.

        Args:
            pred (str): The predicted label.
            confidence (float): The confidence score of the prediction.
            one_meta (dict): The metadata for the current input.
            now (float): The current timestamp.
            ensemble_size (int, optional): The size of the ensemble. Defaults to None.
        """
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

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        """
        Perform inference on incoming samples.

        This method must be implemented by concrete classifiers.
        The scheduler periodically invokes it with new data.

        Parameters
        ----------
        samples : dict[str, dict[int, numpy.ndarray]]
            Nested mapping of time-series data.

            - Keys (``str``): Topic name, such as ``rs1/mic`` or ``rs1/geo``.
            - Values (``dict``): Data received from that topic.
                - Keys (``int``): Unix timestamp in seoncds.
                - Values (``numpy.ndarray``): One-dimensional arrays containing the raw samples
                  recorded during that second.

            Example
            -------
            .. code-block:: python

                samples = {
                    "rs1-1/geo": {
                        1757515699: array([16961, 16907, ..., 16551, 16370]),
                        1757515700: array([16217, 16418, ..., 16913, 16799]),
                    },
                    "rs-1/mic": {
                        1757515699: array([-62, -51, -63, ..., 182, 207, 178]),
                        1757515700: array([194, 210, 174, ..., 233, 247, 210]),
                    },
                }

        Returns
        -------
        Any
            Inference results produced by the classifier.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
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
        """Handle incoming messages from the message queue."""
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
        """Log the activation status."""
        if self.service_states.get('deactivated', False):
            logger.debug('currently deactivated, standing by')

    def run(self):
        """Run the main loop."""
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
