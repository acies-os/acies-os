import logging
from pathlib import Path

import click
import numpy as np
import torch
from acies.deepsense_augmented.inference import Inference
from acies.deepsense_augmented.input_utils.yaml_utils import load_yaml
from acies.node.logging import init_logger
from acies.node.net import common_options, get_zconf
from acies.vehicle_classifier.base import Classifier
from acies.vehicle_classifier.utils import count_elements

logger = logging.getLogger('acies.infer')


class DeepSense(Classifier):
    def __init__(self, model_config, classifier_config_file, *args, **kwargs):
        self.model_config = load_yaml(model_config)
        super().__init__(classifier_config_file, *args, **kwargs)

    def load_model(self, classifier_config_file: Path):
        model = Inference(classifier_config_file, self.model_config, 'cpu')
        logger.info(
            f'loaded model to cpu, '
            f'definition from {Inference.__name__}, '
            f'weights from {classifier_config_file}, '
            f'#params={len(list(model.classifier.parameters()))}, '
            f'#elements={count_elements(model.classifier)}'
        )
        self.modalities = self.model_config['loc_modalities']['shake']
        _mapping = {'seismic': 'geo', 'acoustic': 'mic', 'audio': 'mic', 'sei': 'geo', 'aco': 'mic'}
        self.modalities = [_mapping[x] for x in self.modalities]
        return model

    def segment_signal(self, signal, window_length, overlap_length):
        segments = []
        start = 0
        while start <= len(signal) - window_length:
            end = start + window_length
            segments.append(signal[start:end])
            start = start + window_length - overlap_length
        segments = torch.stack(segments, dim=0)
        return segments

    def infer(self, samples: dict[str, dict[int, np.ndarray]]):
        arrays = {k: self.concat(v) for k, v in samples.items()}
        arrays = {k.split('/')[1]: v for k, v in arrays.items()}
        seismic_data = arrays['geo']
        acoustic_data = arrays['mic']

        seismic_data = seismic_data
        acoustic_data = acoustic_data

        seismic_data = torch.from_numpy(seismic_data)
        acoustic_data = torch.from_numpy(acoustic_data)

        seismic_data = torch.unsqueeze(seismic_data, -1)  # [200, 1]
        seismic_data = self.segment_signal(seismic_data, 20, 0)  # [10, 20, 1]
        seismic_data = torch.permute(torch.abs(torch.fft.fft(seismic_data)), [2, 0, 1])  # [1, 10, 20]
        seismic_data = torch.unsqueeze(seismic_data, 0)  # [1, 1, 10, 20]

        acoustic_data = torch.unsqueeze(acoustic_data, -1)  # [16000, 1]
        acoustic_data = self.segment_signal(acoustic_data, 1600, 0)  # [10, 1600, 1]
        acoustic_data = torch.permute(torch.abs(torch.fft.fft(acoustic_data)), [2, 0, 1])  # [1, 10, 1600]
        acoustic_data = torch.unsqueeze(acoustic_data, 0)  # [1, 1, 10, 1600]

        data = {'shake': {'audio': acoustic_data, 'seismic': seismic_data}}

        logit = self.model.infer(data)  # returns logits [[x, y, z, w]],
        logit = logit.detach().numpy()

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
@click.option('--model-config', help='Model config yaml.', type=click.Path(exists=True))
def main(mode, connect, listen, topic, namespace, proc_name, deactivated, weight, model_config):
    init_logger(f'{namespace}_{proc_name}.log', get_logger='acies')
    z_conf = get_zconf(mode, connect, listen)

    # initialize the class
    clf = DeepSense(
        model_config=model_config,
        classifier_config_file=weight,
        conf=z_conf,
        mode=mode,
        connect=connect,
        listen=listen,
        topic=topic,
        namespace=namespace,
        proc_name=proc_name,
        deactivated=deactivated,
    )

    # start
    clf.start()
