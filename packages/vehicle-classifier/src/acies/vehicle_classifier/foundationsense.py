import asyncio
import json
from collections import deque
from typing import Dict, List

import click
import numpy as np
import torch
from acies.FoundationSense.inference import ModelForInference
from acies.node import Node, common_options, logger
from acies.vehicle_classifier.utils import (
    DistInference,
    TimeProfiler,
    calculate_mean_energy,
    classification_msg,
    distance_msg,
    get_time_range,
    normalize_key,
    update_sys_argv,
)


class FoundationSenseClassifier(Node):
    def __init__(self, weight, *args, **kwargs):
        # pass other args to parent type
        super().__init__(*args, **kwargs)

        # your inference model
        self.model = self.load_model(weight)

        # buffer incoming messages
        self.buffs = {'sei': deque(), 'aco': deque()}

        # how many input messages the model needs to run inference once
        # each message contains 1s of data:
        #     seismic  :    200 samples
        #     acoustic : 16_000 samples
        self.input_len = 2

        # the topic we publish inference results to
        self.model_name = 'fsense'
        self.pub_topic_vehicle = f'{self.get_hostname()}/{self.model_name}/vehicle'

        # the topic we publish target distance results to
        self.pub_topic_distance = f'{self.get_hostname()}/{self.model_name}/distance'

        # distance classifier
        self.distance_classifier = DistInference()

        # Energy level calculation
        self.acoustic_energy_buffer = []  # Buffer for energy level for acoustic signal
        self.acoustic_energy_buffer_size = 2  # Maximum enegy level buffer size for acoustic signal

        self.seismic_energy_buffer = []  # Buffer for energy level for seismic signal
        self.seismic_energy_buffer_size = 2  # Maximum enegy level buffer size for seismic signal

    def segment_signal(self, signal, window_length, overlap_length):
        segments = []
        start = 0
        while start <= len(signal) - window_length:
            end = start + window_length
            segments.append(signal[start:end])
            start = start + window_length - overlap_length
        segments = torch.stack(segments, dim=0)
        return segments

    def load_model(self, path_to_weight: str):
        """Load model from give path."""

        logger.info(f'{ModelForInference} load model from {path_to_weight}')

        # load weight and initialize your model
        model = ModelForInference(path_to_weight)
        return model

    def compute_husky_score(self, predictions):
        no_car_cf = predictions[0]

        no_car_threshold = 0.70
        vehicle_threshold = 0.60

        husky_cf = 0.0000

        if no_car_cf < no_car_threshold:
            # there is a car
            vehicle_predictions = predictions[1:]

            # polarius,

            # get a list of detected car with confidence score above the threshold
            classified_vehicles = [vec for vec in vehicle_predictions if vec >= vehicle_threshold]

            if len(classified_vehicles) == 0:
                # its likely that husky is there
                husky_cf = 1 - np.mean(vehicle_predictions)

        return husky_cf

    def inference(self):
        # buffer incoming messages
        for k, q in self.queue.items():
            if not q.empty():
                logger.debug(f'enqueue: {k}')
                data = q.get(False)
                data = json.loads(data)
                mod, data = normalize_key(data)
                self.buffs[mod].append(data)

        # publish distance info
        if len(self.buffs['sei']) >= 1 and len(self.buffs['aco']) >= 1:
            # access data without taking it out of the queue
            input_sei = self.buffs['sei'][-1]
            # access data without taking it out of the queue
            input_aco = self.buffs['aco'][-1]
            dist_input = {'x_sei': input_sei['samples'], 'x_aud': input_aco['samples']}
            dist: int = self.distance_classifier.predict_distance(dist_input)
            dist_msg = distance_msg(input_sei['timestamp'], self.model_name, dist)
            self.publish(self.pub_topic_distance, json.dumps(dist_msg))

            # 2. Calcualte current energy levels, update energy bufferes
            sei_energy, self.seismic_energy_buffer = calculate_mean_energy(
                input_sei['samples'],
                self.seismic_energy_buffer,
                self.seismic_energy_buffer_size,
            )
            aco_energy, self.acoustic_energy_buffer = calculate_mean_energy(
                input_aco['samples'],
                self.acoustic_energy_buffer,
                self.acoustic_energy_buffer_size,
            )

        # check if we have enough data to run inference
        if len(self.buffs['sei']) >= self.input_len and len(self.buffs['aco']) >= self.input_len:
            input_sei: List[Dict] = [self.buffs['sei'].popleft() for _ in range(self.input_len)]
            input_aco: List[Dict] = [self.buffs['aco'].popleft() for _ in range(self.input_len)]

            start_time, end_time = get_time_range(input_sei)

            # flatten
            input_sei = np.array([x['samples'] for x in input_sei]).flatten()
            input_aco = np.array([x['samples'] for x in input_aco]).flatten()
            assert len(input_sei) == 200 * self.input_len, f'input_sei={len(input_sei)}'
            assert len(input_aco) == 16000 * self.input_len, f'input_aco={len(input_aco)}'

            # down sampling
            input_sei = input_sei[::2]
            input_aco = input_aco[::2]
            assert len(input_sei) == 100 * self.input_len, f'input_sei={len(input_sei)}'
            assert len(input_aco) == 8000 * self.input_len, f'input_aco={len(input_aco)}'

            # Convert to tensor from numpy
            input_sei = torch.from_numpy(input_sei).float()
            input_aco = torch.from_numpy(input_aco).float()

            input_sei = torch.unsqueeze(input_sei, -1)  # [200, 1]
            input_sei = self.segment_signal(input_sei, 20, 0)  # [10, 20, 1]
            input_sei = torch.permute(torch.abs(torch.fft.fft(input_sei)), [2, 0, 1])  # [1, 10, 20]
            input_sei = torch.unsqueeze(input_sei, 0)  # [1, 1, 10, 20]

            input_aco = torch.unsqueeze(input_aco, -1)  # [16000, 1]
            input_aco = self.segment_signal(input_aco, 1600, 0)  # [10, 1600, 1]
            input_aco = torch.permute(torch.abs(torch.fft.fft(input_aco)), [2, 0, 1])  # [1, 10, 1600]
            input_aco = torch.unsqueeze(input_aco, 0)  # [1, 1, 10, 1600]

            data = {
                'shake': {
                    'audio': input_aco,
                    'seismic': input_sei,
                }
            }

            with TimeProfiler() as timer:
                logits = self.model(data)[0]
                predictions = logits.tolist()
                class_names = self.model.args.dataset_config['vehicle_classification']['class_names']

                husky_cf = self.compute_husky_score(predictions)
                result = {class_names[i].lower(): round(predictions[i], 6) for i in range(len(predictions))}
                # del(result["no_vehicle"])
                result['husky'] = 0.0

            logger.debug(f'Inference time: {timer.elapsed_time_ns / 1e6} ms')

            msg = classification_msg(start_time, end_time, self.model_name, result, sei_energy, aco_energy)
            logger.info(f'{self.pub_topic_vehicle}: {msg}')
            self.publish(self.pub_topic_vehicle, json.dumps(msg))

    async def run(self):
        try:
            # Register your inference func as a callback.
            await self.add_callback(self.inference)

            # You can add more callbacks if needed, they will be run concurrently.
            # await self.add_callback(self.another_callback)
            # await self.add_callback(self.3rd_callback)

            # self.start() must be called
            await self.start()
        except KeyboardInterrupt:
            self.close()


@click.command(context_settings=dict(ignore_unknown_options=True))
@common_options
@click.option(
    '-w',
    '--weight',
    help='Model weight',
    type=str,
)
@click.argument('model_args', nargs=-1, type=click.UNPROCESSED)
def main(mode, connect, listen, key, weight, model_args):
    # let the node swallows the args that it needs,
    # and passes the rest to the neural network model
    update_sys_argv(model_args)

    # initialize the class
    classifier = FoundationSenseClassifier(
        mode=mode,
        connect=connect,
        listen=listen,
        sub_keys=key,
        pub_keys=[],
        weight=weight,
    )
    asyncio.run(classifier.run())
