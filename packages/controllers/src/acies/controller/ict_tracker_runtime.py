from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def _parse_float(value: str) -> float:
    if value == '' or value.lower() == 'nan':
        return float('nan')
    return float(value)


def _parse_int(value: str) -> int:
    return int(float(value))


def _zscore_array(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    std = float(np.std(values))
    if std == 0.0 or math.isnan(std):
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _softmax(values: npt.NDArray[np.float64], temperature: float = 1.0) -> npt.NDArray[np.float64]:
    scaled = values * temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    total = np.sum(exp)
    if total <= 0.0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / total


def signed_path_delta(src_idx: int, dst_idx: int, n_nodes: int, is_loop: bool) -> int:
    raw = dst_idx - src_idx
    if not is_loop:
        return raw
    half = n_nodes // 2
    if raw > half:
        raw -= n_nodes
    elif raw < -half:
        raw += n_nodes
    return raw


@dataclass(frozen=True)
class FeatureNormalizer:
    feature_names: tuple[str, ...]
    mean: npt.NDArray[np.float64]
    std: npt.NDArray[np.float64]

    def normalize(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = values.astype(np.float64, copy=True)
        for idx, value in enumerate(out):
            if np.isnan(value):
                continue
            denom = self.std[idx]
            if denom <= 0.0 or np.isnan(denom):
                out[idx] = 0.0
            else:
                out[idx] = (value - self.mean[idx]) / denom
        return out

    @classmethod
    def identity(cls, feature_names: list[str]) -> FeatureNormalizer:
        count = len(feature_names)
        return cls(
            feature_names=tuple(feature_names),
            mean=np.zeros(count, dtype=np.float64),
            std=np.ones(count, dtype=np.float64),
        )


@dataclass(frozen=True)
class StationTemplate:
    station_id: int
    side_label: str
    features: npt.NDArray[np.float64]


@dataclass(frozen=True)
class NodeRecord:
    node_index: int
    sensor_node: str
    point_index: int
    station_id: int
    side_label: str
    sensor_rank: int
    latitude: float
    longitude: float
    x_m: float
    y_m: float


@dataclass(frozen=True)
class ContinuityRuntimeConfig:
    topology: str
    modality: str
    feature_names: tuple[str, ...]
    lag_steps: int = 5
    template_weight: float = 1.8
    anchor_weight: float = 1.0
    neighbor_anchor_weight: float = 0.55
    max_step_nodes: int = 2
    hop_penalty: float = 0.7
    direction_bonus: float = 0.35
    direction_penalty: float = 0.25
    stay_penalty: float = 0.05
    reset_penalty: float = 4.0
    margin_scale: float = 1.0
    change_margin_threshold: float = 0.75
    hybrid_margin_threshold: float = 0.70
    direction_window: int = 5
    direction_threshold: float = 0.08
    anchor_temperature: float = 1.25
    lock_run_direction: bool = False

    @property
    def is_loop(self) -> bool:
        return self.topology == 'loop'


@dataclass(frozen=True)
class DeploymentAssets:
    config: ContinuityRuntimeConfig
    feature_normalizer: FeatureNormalizer
    sensor_order: tuple[str, ...]
    sensor_to_station: dict[str, int]
    sensor_to_cross: dict[str, float]
    station_nodes: dict[int, tuple[str, ...]]
    station_side_templates: tuple[StationTemplate, ...]
    station_side_states: tuple[tuple[int, str], ...]
    station_side_template_matrix: npt.NDArray[np.float64]
    continuity_templates: npt.NDArray[np.float64]
    lattice_nodes: tuple[NodeRecord, ...]
    edges_by_dst: dict[int, tuple[int, ...]]
    sensor_to_rank: dict[str, int]
    station_side_to_sensor: dict[tuple[int, str], str]

    @classmethod
    def load(cls, asset_dir: str | Path) -> DeploymentAssets:
        root = Path(asset_dir)
        metadata = json.loads((root / 'metadata.json').read_text())
        feature_names = [str(name) for name in metadata['feature_names']]
        runtime_cfg = metadata.get('runtime', {})
        config = ContinuityRuntimeConfig(
            topology=str(metadata.get('topology', 'loop')),
            modality=str(metadata.get('modality', 'mic')),
            feature_names=tuple(feature_names),
            lag_steps=int(runtime_cfg.get('lag_steps', 5)),
            template_weight=float(runtime_cfg.get('template_weight', 1.8)),
            anchor_weight=float(runtime_cfg.get('anchor_weight', 1.0)),
            neighbor_anchor_weight=float(runtime_cfg.get('neighbor_anchor_weight', 0.55)),
            max_step_nodes=int(runtime_cfg.get('max_step_nodes', 2)),
            hop_penalty=float(runtime_cfg.get('hop_penalty', 0.7)),
            direction_bonus=float(runtime_cfg.get('direction_bonus', 0.35)),
            direction_penalty=float(runtime_cfg.get('direction_penalty', 0.25)),
            stay_penalty=float(runtime_cfg.get('stay_penalty', 0.05)),
            reset_penalty=float(runtime_cfg.get('reset_penalty', 4.0)),
            margin_scale=float(runtime_cfg.get('margin_scale', 1.0)),
            change_margin_threshold=float(runtime_cfg.get('change_margin_threshold', 0.75)),
            hybrid_margin_threshold=float(runtime_cfg.get('hybrid_margin_threshold', 0.70)),
            direction_window=int(runtime_cfg.get('direction_window', 5)),
            direction_threshold=float(runtime_cfg.get('direction_threshold', 0.08)),
            anchor_temperature=float(runtime_cfg.get('anchor_temperature', 1.25)),
            lock_run_direction=bool(runtime_cfg.get('lock_run_direction', False)),
        )

        normalizer = cls._load_normalizer(root / 'feature_stats.csv', feature_names)
        sensor_rows = cls._load_rows(root / 'sensor_geometry.csv')
        lattice_rows = cls._load_rows(root / 'lattice_points.csv')
        template_rows = cls._load_rows(root / 'station_side_templates.csv')
        continuity_rows = cls._load_rows(root / 'continuity_templates.csv')
        edge_rows = cls._load_rows(root / 'lattice_edges.csv')

        sensor_order = tuple(str(node) for node in metadata.get('sensor_order', []))
        if not sensor_order:
            sensor_order = tuple(
                str(row['node'])
                for row in sorted(
                    sensor_rows, key=lambda row: (_parse_int(row['station_id']), _parse_float(row['cross_m']))
                )
            )

        sensor_to_station = {str(row['node']): _parse_int(row['station_id']) for row in sensor_rows}
        sensor_to_cross = {str(row['node']): _parse_float(row['cross_m']) for row in sensor_rows}
        station_nodes_dict: dict[int, list[tuple[float, str]]] = defaultdict(list)
        for row in sensor_rows:
            station_nodes_dict[_parse_int(row['station_id'])].append((_parse_float(row['cross_m']), str(row['node'])))
        station_nodes = {
            station_id: tuple(node for _cross, node in sorted(rows)) for station_id, rows in station_nodes_dict.items()
        }
        station_side_to_sensor = {
            (station_id, 'negative_cross'): nodes[0] for station_id, nodes in station_nodes.items()
        }
        station_side_to_sensor.update(
            {(station_id, 'positive_cross'): nodes[-1] for station_id, nodes in station_nodes.items()}
        )
        sensor_to_rank = {sensor: idx for idx, sensor in enumerate(sensor_order)}

        station_side_templates = tuple(
            StationTemplate(
                station_id=_parse_int(row['station_id']),
                side_label=str(row['side_label']),
                features=np.array([_parse_float(row[name]) for name in feature_names], dtype=np.float64),
            )
            for row in template_rows
        )
        station_side_states = tuple((template.station_id, template.side_label) for template in station_side_templates)
        station_side_template_matrix = np.vstack([template.features for template in station_side_templates])

        continuity_rows = sorted(continuity_rows, key=lambda row: _parse_int(row['loop_node_index']))
        continuity_templates = np.vstack(
            [np.array([_parse_float(row[name]) for name in feature_names], dtype=np.float64) for row in continuity_rows]
        )

        lattice_nodes = tuple(
            NodeRecord(
                node_index=_parse_int(row['loop_node_index']),
                sensor_node=str(row['sensor_node']),
                point_index=_parse_int(row['point_index']),
                station_id=_parse_int(row['station_id']),
                side_label=str(row['side_label']),
                sensor_rank=_parse_int(row['sensor_rank']),
                latitude=_parse_float(row['latitude']),
                longitude=_parse_float(row['longitude']),
                x_m=_parse_float(row['x_m']),
                y_m=_parse_float(row['y_m']),
            )
            for row in sorted(lattice_rows, key=lambda row: _parse_int(row['loop_node_index']))
        )
        edges_by_dst: dict[int, list[int]] = defaultdict(list)
        for row in edge_rows:
            edges_by_dst[_parse_int(row['dst_loop_node_index'])].append(_parse_int(row['src_loop_node_index']))
        edge_lookup = {dst: tuple(sorted(srcs)) for dst, srcs in edges_by_dst.items()}
        for node_idx in range(len(lattice_nodes)):
            edge_lookup.setdefault(node_idx, tuple())

        return cls(
            config=config,
            feature_normalizer=normalizer,
            sensor_order=sensor_order,
            sensor_to_station=sensor_to_station,
            sensor_to_cross=sensor_to_cross,
            station_nodes=station_nodes,
            station_side_templates=station_side_templates,
            station_side_states=station_side_states,
            station_side_template_matrix=station_side_template_matrix,
            continuity_templates=continuity_templates,
            lattice_nodes=lattice_nodes,
            edges_by_dst=edge_lookup,
            sensor_to_rank=sensor_to_rank,
            station_side_to_sensor=station_side_to_sensor,
        )

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, str]]:
        with path.open('r', encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _load_normalizer(path: Path, feature_names: list[str]) -> FeatureNormalizer:
        if not path.exists():
            return FeatureNormalizer.identity(feature_names)
        stats: dict[str, tuple[float, float]] = {}
        with path.open('r', encoding='utf-8', newline='') as handle:
            for row in csv.DictReader(handle):
                stats[str(row['feature'])] = (_parse_float(row['mean']), max(_parse_float(row['std']), 1e-9))
        mean = np.array([stats.get(name, (0.0, 1.0))[0] for name in feature_names], dtype=np.float64)
        std = np.array([stats.get(name, (0.0, 1.0))[1] for name in feature_names], dtype=np.float64)
        return FeatureNormalizer(feature_names=tuple(feature_names), mean=mean, std=std)


@dataclass
class HybridMicState:
    prev_station: int | None = None
    centroid_history: deque[float] = field(default_factory=deque)
    direction_history: deque[float] = field(default_factory=deque)


@dataclass(frozen=True)
class HybridMicObservation:
    station_id: int
    side_label: str
    direction_label: str
    station_margin: float
    station_centroid: float
    source_mode: str


@dataclass(frozen=True)
class RuntimeStepOutput:
    sample_timestamp_ns: int
    finalize_timestamp_ns: int
    loop_node_index: int
    station_id: int
    side_label: str
    latitude: float
    longitude: float
    x_m: float
    y_m: float
    direction_label: str
    station_margin: float
    confidence: float
    reset: bool
    label: str | None = None


@dataclass
class FixedLagDecoderState:
    dp: list[npt.NDArray[np.float64]] = field(default_factory=list)
    back: list[npt.NDArray[np.int_]] = field(default_factory=list)
    reset_flags: list[npt.NDArray[np.bool_]] = field(default_factory=list)
    observations: list[HybridMicObservation] = field(default_factory=list)
    sample_timestamps_ns: list[int] = field(default_factory=list)
    committed: list[int] = field(default_factory=list)


class HybridMicRuntime:
    def __init__(self, assets: DeploymentAssets):
        self.assets = assets
        self.state = HybridMicState()

    def step(self, normalized_features: npt.NDArray[np.float64]) -> HybridMicObservation:
        raw_station, raw_margin, station_centroid = self._mic_station_estimate(normalized_features)
        station_id = self._apply_station_inertia(raw_station, raw_margin)
        raw_side = self._station_side_from_mic(normalized_features, station_id)
        template_station, template_side, template_margin = self._template_fallback(normalized_features)

        if raw_margin < self.assets.config.hybrid_margin_threshold:
            station_id = template_station
            side_label = template_side
            source_mode = 'template_fallback'
            station_margin = template_margin
        else:
            side_label = raw_side
            source_mode = 'pair_mic'
            station_margin = raw_margin

        direction_label = self._direction_label(station_centroid)
        return HybridMicObservation(
            station_id=station_id,
            side_label=side_label,
            direction_label=direction_label,
            station_margin=station_margin,
            station_centroid=station_centroid,
            source_mode=source_mode,
        )

    def _mic_station_estimate(self, features: npt.NDArray[np.float64]) -> tuple[int, float, float]:
        station_scores: dict[int, float] = {}
        for station_id, nodes in self.assets.station_nodes.items():
            pair_scores = [self._feature_for_sensor(features, node) for node in nodes]
            finite = [score for score in pair_scores if np.isfinite(score)]
            station_scores[station_id] = max(finite) if finite else -1e9

        ordered_station_ids = sorted(station_scores)
        ordered_scores = np.array([station_scores[station_id] for station_id in ordered_station_ids], dtype=np.float64)
        weights = _softmax(ordered_scores, temperature=self.assets.config.anchor_temperature)
        centroid = float(np.sum(np.array(ordered_station_ids, dtype=np.float64) * weights))

        ranked = sorted(station_scores.items(), key=lambda item: item[1], reverse=True)
        raw_station = int(ranked[0][0])
        second_score = ranked[1][1] if len(ranked) > 1 else ranked[0][1]
        return raw_station, float(ranked[0][1] - second_score), centroid

    def _apply_station_inertia(self, raw_station: int, raw_margin: float) -> int:
        prev_station = self.state.prev_station
        if prev_station is None:
            station_id = raw_station
        elif raw_station == prev_station:
            station_id = raw_station
        elif abs(raw_station - prev_station) > 1:
            station_id = prev_station + (1 if raw_station > prev_station else -1)
        elif raw_margin < self.assets.config.change_margin_threshold:
            station_id = prev_station
        else:
            station_id = raw_station
        self.state.prev_station = station_id
        return station_id

    def _station_side_from_mic(self, features: npt.NDArray[np.float64], station_id: int) -> str:
        nodes = self.assets.station_nodes[station_id]
        best_sensor = nodes[0]
        best_score = -1e9
        for sensor in nodes:
            score = self._feature_for_sensor(features, sensor)
            score = score if np.isfinite(score) else -1e9
            if score > best_score:
                best_score = score
                best_sensor = sensor
        return 'positive_cross' if self.assets.sensor_to_cross[best_sensor] >= 0.0 else 'negative_cross'

    def _template_fallback(self, features: npt.NDArray[np.float64]) -> tuple[int, str, float]:
        residual = self.assets.station_side_template_matrix - features[None, :]
        dists = np.nanmean(np.square(residual), axis=1)
        if np.all(np.isnan(dists)):
            state = self.assets.station_side_states[0]
            return state[0], state[1], 0.0
        nan_fill = np.nanmax(dists[np.isfinite(dists)]) + 1.0 if np.isfinite(dists).any() else 1.0
        dists = np.where(np.isnan(dists), nan_fill, dists)
        ranked_idx = np.argsort(dists)
        best_idx = int(ranked_idx[0])
        second_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else best_idx
        state = self.assets.station_side_states[best_idx]
        margin = float(dists[second_idx] - dists[best_idx])
        return int(state[0]), str(state[1]), margin

    def _direction_label(self, centroid: float) -> str:
        history = self.state.centroid_history
        history.append(centroid)
        max_len = max(2, self.assets.config.direction_window)
        while len(history) > max_len:
            history.popleft()
        if len(history) < 2:
            return 'ambiguous'
        diffs = np.diff(np.array(history, dtype=np.float64))
        median_delta = float(np.median(diffs))
        if median_delta >= self.assets.config.direction_threshold:
            return 'toward_S4'
        if median_delta <= -self.assets.config.direction_threshold:
            return 'toward_S1'
        return 'ambiguous'

    def _feature_for_sensor(self, features: npt.NDArray[np.float64], sensor: str) -> float:
        for idx, feature_name in enumerate(self.assets.config.feature_names):
            sensor_name, _sep, _modality = feature_name.partition('__')
            if sensor_name == sensor:
                return float(features[idx])
        return float('nan')


class FixedLagContinuityRuntime:
    def __init__(self, assets: DeploymentAssets):
        self.assets = assets
        self.hybrid = HybridMicRuntime(assets)
        self.decoder = FixedLagDecoderState()
        self.run_direction_sign = 0

    def reset(self) -> None:
        self.hybrid = HybridMicRuntime(self.assets)
        self.decoder = FixedLagDecoderState()
        self.run_direction_sign = 0

    def step(
        self,
        raw_features: npt.NDArray[np.float64],
        sample_timestamp_ns: int,
        finalize_timestamp_ns: int,
        label: str | None = None,
        normalizer_override: FeatureNormalizer | None = None,
    ) -> list[RuntimeStepOutput]:
        normalizer = normalizer_override or self.assets.feature_normalizer
        normalized = normalizer.normalize(raw_features)
        observation = self.hybrid.step(normalized)
        emission = self._build_emission(normalized, observation)
        self._advance_decoder(observation, emission, sample_timestamp_ns)

        outputs: list[RuntimeStepOutput] = []
        commit_idx = len(self.decoder.sample_timestamps_ns) - 1 - self.assets.config.lag_steps
        if commit_idx >= 0 and self.decoder.committed[commit_idx] < 0:
            outputs.append(self._commit_index(commit_idx, finalize_timestamp_ns, label))
        return outputs

    def flush(self, finalize_timestamp_ns: int, label: str | None = None) -> list[RuntimeStepOutput]:
        outputs: list[RuntimeStepOutput] = []
        for idx, state in enumerate(self.decoder.committed):
            if state >= 0:
                continue
            outputs.append(self._commit_index(idx, finalize_timestamp_ns, label))
        return outputs

    def _build_emission(
        self,
        normalized: npt.NDArray[np.float64],
        observation: HybridMicObservation,
    ) -> npt.NDArray[np.float64]:
        residual = self.assets.continuity_templates - normalized[None, :]
        dists = np.nanmean(np.square(residual), axis=1)
        if np.all(np.isnan(dists)):
            template_score = np.zeros(len(self.assets.lattice_nodes), dtype=np.float64)
        else:
            nan_fill = np.nanmax(dists[np.isfinite(dists)]) + 1.0 if np.isfinite(dists).any() else 1.0
            dists = np.where(np.isnan(dists), nan_fill, dists)
            template_score = _zscore_array(-dists)

        anchor_sensor = self.assets.station_side_to_sensor[(observation.station_id, observation.side_label)]
        anchor_rank = self.assets.sensor_to_rank[anchor_sensor]
        anchor_score = self._anchor_score_vector(anchor_rank, observation.station_margin)
        return self.assets.config.template_weight * template_score + anchor_score

    def _anchor_score_vector(self, anchor_sensor_rank: int, margin: float) -> npt.NDArray[np.float64]:
        n_sensors = max(len(self.assets.sensor_order), 1)
        margin_trust = max(0.2, min(1.5, float(margin) / max(self.assets.config.margin_scale, 1e-6)))

        scores = np.zeros(len(self.assets.lattice_nodes), dtype=np.float64)
        for idx, node in enumerate(self.assets.lattice_nodes):
            dist = abs(node.sensor_rank - anchor_sensor_rank)
            if self.assets.config.is_loop:
                dist = min(dist, n_sensors - dist)
            if dist == 0:
                score = self.assets.config.anchor_weight * margin_trust
            elif dist == 1:
                score = self.assets.config.neighbor_anchor_weight * (2.0 - margin_trust)
            else:
                score = -float(dist) * margin_trust
            scores[idx] = score
        return scores

    def _advance_decoder(
        self,
        observation: HybridMicObservation,
        emission: npt.NDArray[np.float64],
        sample_timestamp_ns: int,
    ) -> None:
        state = self.decoder
        n_nodes = len(self.assets.lattice_nodes)
        state.observations.append(observation)
        state.sample_timestamps_ns.append(sample_timestamp_ns)
        state.committed.append(-1)

        if not state.dp:
            state.dp.append(emission.copy())
            state.back.append(np.full(n_nodes, -1, dtype=int))
            state.reset_flags.append(np.ones(n_nodes, dtype=bool))
            return

        prev_scores = state.dp[-1]
        row_scores = np.full(n_nodes, -1e18, dtype=np.float64)
        row_back = np.full(n_nodes, -1, dtype=int)
        row_reset = np.ones(n_nodes, dtype=bool)
        expected_sign = self._expected_direction_sign(observation)

        all_sources = range(n_nodes)
        for dst in range(n_nodes):
            best_score = float(emission[dst] - self.assets.config.reset_penalty)
            best_src = -1
            best_reset = True
            for src in all_sources:
                delta = signed_path_delta(src, dst, n_nodes=n_nodes, is_loop=self.assets.config.is_loop)
                if abs(delta) > self.assets.config.max_step_nodes:
                    continue
                if self.run_direction_sign != 0 and delta != 0 and int(np.sign(delta)) != self.run_direction_sign:
                    continue
                candidate = float(
                    prev_scores[src] + emission[dst] + self._transition_score(delta=delta, expected_sign=expected_sign)
                )
                if candidate > best_score:
                    best_score = candidate
                    best_src = src
                    best_reset = False
            row_scores[dst] = best_score
            row_back[dst] = best_src
            row_reset[dst] = best_reset

        state.dp.append(row_scores)
        state.back.append(row_back)
        state.reset_flags.append(row_reset)

    def _expected_direction_sign(self, observation: HybridMicObservation) -> int:
        if self.assets.config.lock_run_direction:
            sign = self._global_direction_sign(observation.side_label, observation.direction_label)
            if self.run_direction_sign == 0 and sign != 0:
                self.run_direction_sign = sign
            return self.run_direction_sign
        return self._global_direction_sign(observation.side_label, observation.direction_label)

    @staticmethod
    def _global_direction_sign(side_label: str, direction_label: str) -> int:
        if direction_label == 'ambiguous':
            return 0
        if side_label == 'negative_cross':
            return 1 if direction_label == 'toward_S4' else -1
        return 1 if direction_label == 'toward_S1' else -1

    def _transition_score(self, delta: int, expected_sign: int) -> float:
        if delta == 0:
            if expected_sign != 0:
                # When direction is detected, raise the stay cost to 85% of the net
                # forward hop cost. This removes artificial inertia in dense flat-emission
                # segments without applying the penalty during ambiguous phases (where
                # forcing movement with no directional evidence causes erratic jumps).
                net_forward_cost = self.assets.config.hop_penalty - self.assets.config.direction_bonus
                return -(net_forward_cost * 0.85)
            return -self.assets.config.stay_penalty
        score = -self.assets.config.hop_penalty * abs(delta)
        if expected_sign != 0:
            if int(np.sign(delta)) == expected_sign:
                score += self.assets.config.direction_bonus * abs(delta)
            else:
                score -= self.assets.config.direction_penalty * abs(delta)
        return score

    def _commit_index(self, sample_idx: int, finalize_timestamp_ns: int, label: str | None) -> RuntimeStepOutput:
        best_t = min(sample_idx + self.assets.config.lag_steps, len(self.decoder.dp) - 1)
        horizon_state = int(np.argmax(self.decoder.dp[best_t]))
        state = horizon_state
        for back_t in range(best_t, sample_idx, -1):
            prev = self.decoder.back[back_t][state]
            if prev < 0:
                break
            state = int(prev)

        self.decoder.committed[sample_idx] = state
        node = self.assets.lattice_nodes[state]
        row_scores = self.decoder.dp[best_t]
        best_score = float(row_scores[state])
        if len(row_scores) > 1:
            second = float(np.partition(row_scores, -2)[-2])
            confidence = best_score - second
        else:
            confidence = 0.0
        observation = self.decoder.observations[sample_idx]
        reset = bool(self.decoder.reset_flags[best_t][state])
        return RuntimeStepOutput(
            sample_timestamp_ns=self.decoder.sample_timestamps_ns[sample_idx],
            finalize_timestamp_ns=finalize_timestamp_ns,
            loop_node_index=state,
            station_id=node.station_id,
            side_label=node.side_label,
            latitude=node.latitude,
            longitude=node.longitude,
            x_m=node.x_m,
            y_m=node.y_m,
            direction_label=observation.direction_label,
            station_margin=observation.station_margin,
            confidence=confidence,
            reset=reset,
            label=label,
        )
