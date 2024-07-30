import logging
from pathlib import Path

import click
import numpy as np
import torch
from acies.core import common_options, get_zconf, init_logger
from acies.FoundationSense.inference import ModelForInference
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import TimeProfiler, count_elements, update_sys_argv

logger = logging.getLogger('acies.infer')


class VibroFM(Classifier):
    def __init__(self, modality, classifier_config_file, *args, **kwargs):
        self._single_modality = modality
        super().__init__(classifier_config_file, *args, **kwargs)

    def load_model(self, classifier_config_file: Path):
        freq_mae = True if 'mae' in self.proc_name else False
        model = ModelForInference(classifier_config_file, freq_mae, modality=self._single_modality)

        logger.info(
            f'loaded model to cpu, '
            f'definition from {ModelForInference.__name__}, '
            f'weights from {classifier_config_file}, '
            f'#params={len(list(model.parameters()))}, '
            f'#elements={count_elements(model)}'
        )
        self.modalities = model.args.dataset_config['modality_names']
        _mapping = {'seismic': 'geo', 'acoustic': 'mic', 'audio': 'mic', 'sei': 'geo', 'aco': 'mic'}
        self.modalities = [_mapping[x] for x in self.modalities]
        return model

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        arrays = {k: self.concat(v) for k, v in samples.items()}
        arrays = {k.split('/')[-1]: v for k, v in arrays.items()}

        # data = {'shake': {'audio': acoustic_data, 'seismic': seismic_data}}
        data = {'shake': {}}
        for mod in self.modalities:
            mod_data = arrays[mod]
            if mod == 'geo':
                mod_data = mod_data[::2].reshape(1, 1, 10, 20)
            else:
                mod_data = mod_data[::2].reshape(1, 1, 10, 1600)
            mod_data = torch.from_numpy(mod_data)
            if mod == 'geo':
                data['shake']['seismic'] = mod_data
            else:
                data['shake']['audio'] = mod_data

        with TimeProfiler() as timer:
            logit = self.model(data)  # returns logits [[x, y, z, w]],
        elapsed_ms = timer.elapsed_time_ns / 1e6
        logger.debug(f'Time (ms) to infer: {elapsed_ms}')

        # result = {
        #     "gle350": logit[0][0],
        #     "miata": logit[0][1],
        #     "cx30": logit[0][2],
        #     "mustang": logit[0][3],
        # }
        result = dict(zip(np.arange(4), logit[0]))

        return result


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option('--weight', help='Model weight', type=click.Path(exists=True))
@click.option('--modality', type=str, help='Single modality: seismic, audio')
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
    deactivated,
    weight,
    modality,
    model_args,
    sync_interval,
    feature_twin,
    twin_model,
    twin_buff_len,
    heartbeat_interval_s,
):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    log_file = f'{namespace.replace("/", "_")}_{proc_name.replace("/", "_")}.log'
    init_logger(log_file, name='acies')
    z_conf = get_zconf(mode, connect, listen)

    logger.debug(f'{modality=}')

    # initialize the class
    clf = VibroFM(
        modality=modality,
        conf=z_conf,
        twin_model=twin_model,
        twin_buff_len=twin_buff_len,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        deactivated=deactivated,
        classifier_config_file=weight,
        sync_interval=sync_interval,
        feature_twin=feature_twin,
        heartbeat_interval_s=heartbeat_interval_s,
    )

    # start
    clf.start()
