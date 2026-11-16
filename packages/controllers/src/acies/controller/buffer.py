from collections import defaultdict
from dataclasses import dataclass, field

from acies.core import AciesMsg


@dataclass
class PeerEnsembleBuff:
    _data: dict[str, AciesMsg] = field(default_factory=dict, repr=True)

    def add(self, msg: AciesMsg):
        # # ts is of the form: {'twin/rs10/geo': {1716260262: {...}}}
        # ts = msg.meta['inputs']
        # # use the oldest input message timestamp as the key
        # k = min([int(t) for v in ts.values() for t in v])
        k = msg.reply_to
        self._data[k] = msg

    def ensemble(self, ensemble_win_size: int, ensemble_size: int):
        newest = max([int(v.timestamp / 1e9) for v in self._data.values()])
        oldest = newest - ensemble_win_size
        vals = [v for v in self._data.values() if int(v.timestamp / 1e9) >= oldest]
        if len(vals) >= ensemble_size:
            result = self._soft_voting(vals)
            # meta={
            #         'timestamp': datetime.now().timestamp(),
            #         'inference_time_ms': infer_time_ms,
            #         'inputs': dict(meta_data),
            #     },
            ts = max(int(v.timestamp / 1e9) for v in vals)
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
