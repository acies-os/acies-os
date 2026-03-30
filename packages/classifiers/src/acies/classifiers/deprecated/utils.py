import sys
from time import perf_counter_ns
from typing import Dict, List, Tuple

import numpy as np


def calculate_mean_energy(
    sample: List[float], energy_buffer: List[float], buffer_size: int
) -> Tuple[float, List[float]]:
    energy = float(np.std(sample))
    if len(energy_buffer) >= buffer_size:
        energy_buffer.pop(0)
    energy_buffer.append(energy)

    mean_energy = float(np.mean(energy_buffer))
    return mean_energy, energy_buffer


def normalize_key(data: Dict) -> Tuple[str, Dict]:
    assert 'samples' in data
    if 'channel' in data:
        return 'sei', data
    elif 'sample_rate':
        return 'aco', data
    else:
        raise KeyError(f'{data} should contain key: `channel` or `sample_rate`')


def get_time_range(data: List[Dict]) -> Tuple[int, int]:
    start = data[0]['timestamp']
    end = data[-1]['timestamp']
    return start, end


def classification_msg(
    start: int,
    end: int,
    model: str,
    result: Dict[str, float],
    seismic_energy: float,
    acoustic_energy: float,
) -> Dict:
    msg = {
        'start': start,
        'end': end,
        'model': model,
        'result': result,
        'seismic_energy': seismic_energy,
        'acoustic_energy': acoustic_energy,
    }
    return msg


def distance_msg(timestamp: int, model: str, distance: float) -> Dict:
    msg = {'timestamp': timestamp, 'model': model, 'distance': distance}
    return msg


class TimeProfiler:
    def __enter__(self):
        self._start = perf_counter_ns()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed_time_ns = perf_counter_ns() - self._start


def update_sys_argv(model_args: Tuple):
    sys.argv = [sys.argv[0]] + list(model_args)


# Parameters for Acousmic Collection
AUDIO_THRESHOLDS = [3e10, 8.6e10, 35e10, 80e10]
GEO_THRESHOLDS = [5.9e10, 6.25e10, 7.8e10, 9e10]

AUDIO_THRESHOLDS = [x / 16 for x in AUDIO_THRESHOLDS]
GEO_THRESHOLDS = [x / 2 for x in GEO_THRESHOLDS]

PAST_GEO = 2
FUTURE_GEO = 3
INIT_DISTANCE = 0


## detection labels
def create_label_dictionary(size, label_names):
    label_dictionary = {i: label_names[i] for i in range(size)}
    return label_dictionary


class Queue:
    def __init__(self, buff_len):
        self.queue = []
        self.max_len = buff_len

    def push(self, state):
        if len(self.queue) == self.max_len:
            self.queue.pop(0)
        self.queue.append(state)

    def get_state(self):
        return self.queue[-1]

    def get_queue(self):
        return self.queue

    def length(self):
        return len(self.queue)

    def pop(self):
        return self.queue.pop(0)

    def get_majority(self):
        return max(set(self.queue), key=self.queue.count)


class DistInference(object):
    def __init__(
        self,
        model_path=None,
        init_distance=INIT_DISTANCE,
        threshold_audio=AUDIO_THRESHOLDS,
        threshold_geo=GEO_THRESHOLDS,
        past_geo=PAST_GEO,
        future_geo=FUTURE_GEO,
    ):
        """
        Returns discrete distance prediction for a single node

        Usage:
        distance_classifier = DistInference()
        distance = distance_classifier.predict_distance(data)

        Returns: 0,1,2- 0 for far and 2 for close
        """

        ### classifier init
        self.targets = ['Warhog', 'Polaris', 'Silverado']  # , "Husky"
        self.acoustic_energy_threshold = 12e10
        self.seismic_energy_threshold = 6.1e10

        ### distance detector init
        assert past_geo > 1
        assert future_geo > 1
        self.threshold_audio = threshold_audio
        self.threshold_geo = threshold_geo

        self.past_geo = past_geo
        self.future_geo = future_geo
        self.init_distance = init_distance

        self.base_state = 0
        self.past_audio_state_buffer = 0
        self.previous_distance = init_distance
        self.geo_buffer = Queue(self.past_geo + self.future_geo)
        self.audio_buffer = Queue(self.future_geo)
        self.thresholds_audio = []
        self.geo_thresholds = []

    def baseline_detection(self, data):
        # this is for pure detection, i.e. no ML, is there a vehicle or not?
        x_aud = data['x_aud']
        x_sei = data['x_sei']
        # TODO: Mistake, downsampling affects energy calculation
        energy_aud = calculate_energy(x_aud)
        energy_sei = calculate_energy(x_sei)

        if energy_aud > self.acoustic_energy_threshold and energy_sei > self.seismic_energy_threshold:
            return True
        else:
            return False

    #### Distance detector
    def get_audio_state(self, curr_energy):
        diff = np.abs(np.array(self.threshold_audio) - curr_energy)
        min_index = np.argmin(diff)
        return min_index

    def get_geo_state(self, curr_energy):
        # geo_thresholds = [5.5e+10, 6e+10, 6.5e+10, 7.5e+10]
        # round current to nearest threshold

        diff = np.abs(np.array(self.geo_thresholds) - curr_energy)
        min_diff = np.min(diff)
        min_index = np.argmin(diff)
        return min_index

    def predict_distance(self, data):
        # Returns 0,1,2: 0 for far and 2 for close
        x_aud = np.array(data['x_aud'])
        x_sei = np.array(data['x_sei'])
        prediction = self.build_trace(x_aud, x_sei)
        return int(prediction)

    def build_trace(self, packet_audio, packet_geo):
        if self.geo_buffer.length() <= self.past_geo:
            self.geo_buffer.push(packet_geo)
            # print(geo_buffer.get_queue())
            return self.init_distance

        if self.geo_buffer.length() > self.past_geo and self.geo_buffer.length() < self.future_geo + self.past_geo:
            self.geo_buffer.push(packet_geo)
            self.audio_buffer.push(packet_audio)
            self.past_audio_state_buffer = self.get_audio_state(np.sum(packet_audio**2))

            return self.init_distance

        self.geo_buffer.push(packet_geo)

        assert self.geo_buffer.length() == self.past_geo + self.future_geo
        audio_packet_to_predict = self.audio_buffer.pop()
        self.audio_buffer.push(packet_audio)

        geo_queue = self.geo_buffer.get_queue()

        assert len(geo_queue) == self.past_geo + self.future_geo

        audio_energy = np.sum(audio_packet_to_predict**2)

        audio_state = self.get_audio_state(audio_energy)

        # print(audio_state)

        # return

        keep_peak_state = False
        # self.past_buffer_delete.append(past_audio_state_buffer)

        if audio_state == len(self.thresholds_audio) - 1:
            geo_energies = []
            for i in range(self.past_geo + self.future_geo):
                geo_energies.append(np.sum(geo_queue[i] ** 2))

            geo_states = []

            for curr_geo_energy in geo_energies:
                geo_states.append(self.get_geo_state(curr_geo_energy))

            for geo_state in geo_states:
                if geo_state >= len(self.geo_thresholds) - 1:
                    keep_peak_state = True

            # this if condition can be removed and see if its better
            if keep_peak_state == False:
                audio_state = max(geo_states)

        # new_distance = np.sqrt(self.thresholds_audio[self.past_audio_state_buffer]/self.thresholds_audio[audio_state]) * self.previous_distance

        # return new_distance
        return audio_state


def count_elements(model, only_required_grad=False):
    if only_required_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
