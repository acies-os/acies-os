import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from acies.core import AciesMsg

logger = logging.getLogger('acies.infer')


@dataclass
class StreamBuffer:
    size: int
    # {'rs1/mic': {180000: np.ndarray([...]),
    #             },
    # }
    _data: dict[str, dict[int, np.ndarray]] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _meta: dict[str, dict[int, dict]] = field(default_factory=lambda: defaultdict(dict), repr=False)
    _timestamps: Counter[int] = field(default_factory=Counter)

    def add(self, topic: str, timestamp: int, samples: np.ndarray, meta: dict):
        # topic: 'rs1/mic'
        self._data[topic][timestamp] = samples
        self._meta[topic][timestamp] = meta
        self._timestamps[timestamp] += 1

    def _check_size(self):
        while len(self._timestamps) > self.size:
            # find and delete the oldest timestamp
            t = min(self._timestamps.keys())

            for v in self._data.values():
                # v = {t1: np.ndarray([...]), t2: np.ndarray([...])}
                _ = v.pop(t, None)
                self._timestamps[t] -= 1

            for v in self._meta.values():
                # v = {t1: {'timestamp': t1, 'label': l1, ...}, t2: {'timestamp': t2, 'label': l2, ...}}
                _ = v.pop(t, None)

            if self._timestamps[t] == 0:
                del self._timestamps[t]

    def get(self, keys: list[str], n: int) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, dict]]]:
        """Get n-second of samples for all keys."""
        data = defaultdict(dict)
        data_meta = defaultdict(dict)

        for timestamp in sorted(self._timestamps):
            if all(timestamp - i in self._data[k] for k in keys for i in range(n)):
                for k in keys:
                    for i in reversed(range(n)):
                        t = timestamp - i
                        sample = self._data[k].pop(t)
                        sample_meta = self._meta[k].pop(t)

                        self._timestamps[t] -= 1
                        if self._timestamps[t] == 0:
                            del self._timestamps[t]

                        data[k][t] = sample
                        # sample_meta['energy'] = np.std(sample)
                        data_meta[k][t] = sample_meta
                return dict(data), dict(data_meta)
        raise ValueError('not enough data')


@dataclass
class TemporalEnsembleBuff:
    # temporal ensemble buffer size, control how many messages stored in the buffer
    buff_size: int
    _data: dict[int, AciesMsg] = field(default_factory=dict, repr=True)

    def add(self, msg: AciesMsg):
        # ts is of the form: {'twin/rs10/geo': {1716260262: {...}}}
        ts = msg.get_metadata()['inputs']
        # use the oldest input message timestamp as the key
        k = min([int(t) for v in ts.values() for t in v])
        self._data[k] = msg
        self._check_size()

    def _check_size(self):
        while len(self._data) > self.buff_size:
            t = min(self._data.keys())
            del self._data[t]

    def ensemble(self, timestamp_now: int, ensemble_win_size: int, ensemble_size: int):
        oldest = timestamp_now - ensemble_win_size
        vals = [v for k, v in self._data.items() if k >= oldest]
        if len(vals) >= ensemble_size:
            result = self._soft_voting(vals)
            # meta={
            #         'timestamp': datetime.now().timestamp(),
            #         'inference_time_ms': infer_time_ms,
            #         'inputs': dict(meta_data),
            #     },
            # message timestamp is in nanoseconds
            ts = max(v.timestamp / 1e9 for v in vals)
            infer_time_ms = [v.get_metadata()['inference_time_ms'] for v in vals]
            infer_time_ms = sum(infer_time_ms) / len(infer_time_ms)
            inputs = defaultdict(dict)
            for d in vals:
                for k, v in d.get_metadata()['inputs'].items():
                    inputs[k].update(v)
            meta = {
                'timestamp': ts,
                'inference_time_ms': infer_time_ms,
                'inputs': dict(inputs),
                'ensemble_size': len(vals),
            }
            return result, meta
        else:
            raise ValueError(f'not enough data: required {ensemble_size}, got {len(vals)}')

    def _soft_voting(self, preds: list[AciesMsg]):
        result = defaultdict(float)
        for pred in preds:
            for label, logit in pred.get_payload().items():
                result[label] += logit
        total = len(preds)
        assert total >= 0
        for label in result:
            result[label] /= total
        return dict(result)
