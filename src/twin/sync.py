import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import torch

logger = logging.getLogger('acies.dtwin.sync')


class SyncClient:
    def __init__(self, modalities: List[str], *args, **kwargs):
        now = datetime.utcnow()
        self.last_sync_timestamp = {k: now for k in modalities}
        self.last_sync_value = {k: None for k in modalities}
        logger.debug(f'initialized {self}')

    def should_sync(self, x: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def update(self, last_sync_value: dict):
        for k, v in last_sync_value.items():
            self.last_sync_value[k] = v
            self.last_sync_timestamp[k] = datetime.utcnow()

    def update_thresh(self, thresh):
        raise NotImplementedError

    def sync_topic(self, ctrl_topic):
        # return '/'.join(['twin', ctrl_topic])
        return 'controller/ctrl'

    def __repr__(self):
        return f'{self.__class__.__name__}'


class FixedInterval(SyncClient):
    def __init__(self, modalities: List[str], interval_ms: Dict[str, int]):
        super().__init__(modalities)
        self.interval = {k: timedelta(milliseconds=v) for k, v in interval_ms.items()}

    def should_sync(self, x: Dict[str, torch.Tensor]):
        results = [datetime.utcnow() - self.last_sync_timestamp[k] >= self.interval[k] for k in x]
        return any(results)

    def update_thresh(self, thresh):
        pass


class FixedThresh(SyncClient):
    def __init__(self, modalities: List[str], thresh_x: Dict[str, torch.Tensor]):
        super().__init__(modalities)
        self.thresh_x = thresh_x

    def should_sync(self, x: Dict[str, torch.Tensor]):
        results = []
        for k, v in x.items():
            if self.last_sync_value[k] is None:
                return True

            if self.thresh_x[k] is None:
                return True

            delta_x = v - self.last_sync_value[k]
            delta_x = torch.abs(delta_x)
            results.append(torch.any(delta_x >= self.thresh_x))
        return any(results)

    def update_thresh(self, thresh_x):
        for k, v in thresh_x.items():
            if self.thresh_x[k] is None and v is not None:
                logger.debug('initialized thresh_x')
            self.thresh_x[k] = v


class TwinSync(SyncClient):
    def __init__(self, modalities: List[str], thresh: dict):
        super().__init__(modalities)
        self.thresh = thresh

    def should_sync(self, x: Dict[str, torch.Tensor]):
        results = []
        for k, v in x.items():
            if (
                self.thresh[k]['thresh_y'] is None
                or self.thresh[k]['jacobian'] is None
                or self.last_sync_value[k] is None
            ):
                return True

            # for k in self.last_sync_value:
            delta_x = v - self.last_sync_value[k]
            # logger.debug(f'{delta_x=}, {v=}, {self.last_sync_value[k]}')
            delta_y = torch.sum(self.thresh[k]['jacobian'] * delta_x, dim=(1, 2, 3))
            delta_y = torch.abs(delta_y)
            if (delta_y >= self.thresh[k]['thresh_y']).any():
                logger.debug(f'{k=}, delta_y: {delta_y} >= thresh_y')
            results.append((delta_y >= self.thresh[k]['thresh_y']).any())

        return any(results)

    def update_thresh(self, thresh):
        for k, v in thresh.items():
            if self.thresh[k] is None and v is not None:
                logger.debug('initialized thresh')
            self.thresh[k] = v


class SyncServer:
    def __init__(self, error_bound):
        self.error_bound = error_bound
        self.last_sync_value = None
        self.initialized = {
            'fixed_thresh': False,
            'fixed_interval': False,
            'twin_sync': False,
        }

    def update(self, sync_method, state) -> Optional[Dict]:
        thresh = None
        if sync_method == 'twin_sync':
            jacobian = state['jacobian']
            pred = state['pred']
            for mod in pred:
                assert isinstance(jacobian, dict)
                assert isinstance(pred, dict)
                thresh_y = {k: self.error_bound * torch.ones(v.shape) for k, v in pred.items()}
                thresh = {'key': sync_method, 'val': {mod: {'jacobian': jacobian[mod], 'thresh_y': thresh_y[mod]}}}

        elif sync_method == 'fixed_thresh':
            if not self.initialized[sync_method]:
                # if self.last_sync_value is None:
                data = state.pop('twin/tensor')
                data = next(iter(data.values()))
                assert isinstance(data, torch.Tensor)
                thresh = self.error_bound * torch.ones(data.shape)
                thresh = {'key': sync_method, 'val': thresh}
                logger.debug(f'{sync_method} initialized thresh to {thresh}')
                self.initialized[sync_method] = True
        elif sync_method == 'fixed_interval':
            # no need to send a threshold
            pass
        else:
            raise ValueError(f'unknown sync method: {sync_method}')

        self.last_sync_value = state
        return thresh


@dataclass
class EnsembleBuff:
    buff_size: int = field()
    ensemble_size_peer: int = field()
    ensemble_size_temporal: int = field()
    peers: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)
    modalities: list = field(default_factory=list)
    buff: Dict[str, Dict[int, torch.Tensor]] = field(repr=False, default_factory=lambda: defaultdict(dict))

    def add(self, key, timestamp, pred):
        # key:
        #   ns1/n1/acoustic
        #   ns1/n1/seismic
        #   ns2/n2/acoustic
        #   ns2/n2/seismic
        self.buff[key][timestamp] = pred
        node, mod = key.rsplit('/', 1)
        if node not in self.peers:
            self.peers.append(node)
        if mod not in self.modalities:
            self.modalities.append(mod)
        if timestamp not in self.timestamps:
            self.timestamps.append(timestamp)

    def get(self, key: str, timestamp: int):
        return self.buff.get(key, {}).get(timestamp, None)

    def _soft_voting(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        assert isinstance(predictions[0], torch.Tensor)
        result = sum(predictions) / len(predictions)
        assert isinstance(result, torch.Tensor)
        return result

    def ensemble(self, timestamp: int) -> torch.Tensor:
        vals = []
        ts = [timestamp - x for x in range(self.ensemble_size_temporal)]
        for k in self.buff:
            v = [self.buff[k][t] for t in ts if t in self.buff[k]]
            vals.extend(v)
        num_components = self.ensemble_size_temporal * self.ensemble_size_peer
        if len(vals) >= num_components:
            result = self._soft_voting(vals)
            return result
        else:
            raise ValueError(f'not enough components: required {num_components}, got {len(vals)}')

    def ensemble_temporal(self, key: str, timestamp: int) -> torch.Tensor:
        ts = [timestamp - x for x in range(self.ensemble_size_temporal)]
        vals = [self.buff[key][t] for t in ts if t in self.buff[key]]
        if len(vals) >= self.ensemble_size_temporal:
            result = self._soft_voting(vals)
            return result
        else:
            raise ValueError(f'not enough timestamps: required {self.ensemble_size_temporal}, got {len(vals)}')

    def ensemble_peer(self, timestamp: int) -> torch.Tensor:
        peers = self.buff.keys()
        vals = [self.buff[p][timestamp] for p in peers if timestamp in self.buff[p]]
        if len(vals) >= self.ensemble_size_peer:
            result = self._soft_voting(vals)
            return result
        else:
            raise ValueError(f'not enough peers: required {self.ensemble_size_peer}, got {len(vals)}')

    def ensemble_modality(self, keys: List[str], timestamp: int) -> torch.Tensor:
        assert all(k in self.buff for k in keys)
        vals = [self.buff[k][timestamp] for k in keys if timestamp in self.buff[k]]
        if len(vals) > 1:
            result = self._soft_voting(vals)
            return result
        else:
            raise ValueError(f'not enough modalities: required >=2, got {len(vals)}')

    def gc(self):
        """Delete inference results older than `timestamp - self.expire`"""
        if all(len(x) <= self.buff_size for x in self.buff.values()):
            return
        for key in list(self.buff.keys()):
            while len(self.buff[key]) > self.buff_size:
                oldest = min(self.buff[key])
                del self.buff[key][oldest]
        name_mod_list = [x.rsplit('/', 1) for x in self.buff]
        self.peers = list(set([x[0] for x in name_mod_list]))
        self.modalities = list(set([x[1] for x in name_mod_list]))
        self.timestamps = list(set([t for v in self.buff.values() for t in v]))


@dataclass
class EnsembleHistory:
    max_len: int = field()
    history: Dict[int, torch.Tensor] = field(default_factory=dict)

    def add(self, timestamp: int, pred: torch.Tensor):
        self.history[timestamp] = pred
        self.gc()

    def gc(self):
        while len(self.history) > self.max_len:
            oldest = min(self.history)
            del self.history[oldest]
