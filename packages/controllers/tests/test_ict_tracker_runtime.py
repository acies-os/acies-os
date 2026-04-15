from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from acies.controller.ict_tracker_runtime import (
    ContinuityRuntimeConfig,
    DeploymentAssets,
    FeatureNormalizer,
    FixedLagContinuityRuntime,
    NodeRecord,
    StationTemplate,
    signed_path_delta,
)


def _assets() -> DeploymentAssets:
    feature_names = ("n1__mic", "n2__mic", "p2__mic", "p1__mic")
    config = ContinuityRuntimeConfig(
        topology="loop",
        modality="mic",
        feature_names=feature_names,
        lag_steps=2,
        max_step_nodes=1,
        reset_penalty=10.0,
    )
    station_templates = (
        StationTemplate(1, "negative_cross", np.array([1.0, 0.0, 0.0, 0.0])),
        StationTemplate(2, "negative_cross", np.array([0.0, 1.0, 0.0, 0.0])),
        StationTemplate(2, "positive_cross", np.array([0.0, 0.0, 1.0, 0.0])),
        StationTemplate(1, "positive_cross", np.array([0.0, 0.0, 0.0, 1.0])),
    )
    lattice_nodes = (
        NodeRecord(0, "n1", 1, 1, "negative_cross", 0, 0.0, 0.0, 0.0, 0.0),
        NodeRecord(1, "n2", 1, 2, "negative_cross", 1, 0.0, 0.0, 1.0, 0.0),
        NodeRecord(2, "p2", 1, 2, "positive_cross", 2, 0.0, 0.0, 2.0, 0.0),
        NodeRecord(3, "p1", 1, 1, "positive_cross", 3, 0.0, 0.0, 3.0, 0.0),
    )
    return DeploymentAssets(
        config=config,
        feature_normalizer=FeatureNormalizer.identity(list(feature_names)),
        sensor_order=("n1", "n2", "p2", "p1"),
        sensor_to_station={"n1": 1, "n2": 2, "p2": 2, "p1": 1},
        sensor_to_cross={"n1": -1.0, "n2": -1.0, "p2": 1.0, "p1": 1.0},
        station_nodes={1: ("n1", "p1"), 2: ("n2", "p2")},
        station_side_templates=station_templates,
        station_side_states=tuple((row.station_id, row.side_label) for row in station_templates),
        station_side_template_matrix=np.vstack([row.features for row in station_templates]),
        continuity_templates=np.vstack(
            [
                np.array([1.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            ]
        ),
        lattice_nodes=lattice_nodes,
        edges_by_dst={0: (0, 1, 3), 1: (0, 1, 2), 2: (1, 2, 3), 3: (0, 2, 3)},
        sensor_to_rank={"n1": 0, "n2": 1, "p2": 2, "p1": 3},
        station_side_to_sensor={
            (1, "negative_cross"): "n1",
            (2, "negative_cross"): "n2",
            (2, "positive_cross"): "p2",
            (1, "positive_cross"): "p1",
        },
    )


def test_feature_normalizer_identity() -> None:
    normalizer = FeatureNormalizer.identity(["a__mic", "b__mic"])
    values = np.array([1.5, -2.0])
    assert np.allclose(normalizer.normalize(values), values)


def test_fixed_lag_runtime_commits_sequence() -> None:
    runtime = FixedLagContinuityRuntime(_assets())
    feature_rows = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]

    committed = []
    for idx, row in enumerate(feature_rows):
        committed.extend(runtime.step(row, sample_timestamp_ns=idx, finalize_timestamp_ns=idx))
    committed.extend(runtime.flush(finalize_timestamp_ns=10))

    assert [row.loop_node_index for row in committed] == [0, 1, 2, 3]


def test_signed_path_delta_line_does_not_wrap() -> None:
    assert signed_path_delta(3, 0, n_nodes=4, is_loop=False) == -3
    assert signed_path_delta(0, 3, n_nodes=4, is_loop=False) == 3
    assert signed_path_delta(3, 0, n_nodes=4, is_loop=True) == 1


def test_fixed_lag_runtime_line_topology_clamps_boundaries() -> None:
    assets = _assets()
    line_assets = DeploymentAssets(
        config=ContinuityRuntimeConfig(
            topology="line",
            modality=assets.config.modality,
            feature_names=assets.config.feature_names,
            lag_steps=assets.config.lag_steps,
            max_step_nodes=1,
            reset_penalty=100.0,
        ),
        feature_normalizer=assets.feature_normalizer,
        sensor_order=assets.sensor_order,
        sensor_to_station=assets.sensor_to_station,
        sensor_to_cross=assets.sensor_to_cross,
        station_nodes=assets.station_nodes,
        station_side_templates=assets.station_side_templates,
        station_side_states=assets.station_side_states,
        station_side_template_matrix=assets.station_side_template_matrix,
        continuity_templates=assets.continuity_templates,
        lattice_nodes=assets.lattice_nodes,
        edges_by_dst=assets.edges_by_dst,
        sensor_to_rank=assets.sensor_to_rank,
        station_side_to_sensor=assets.station_side_to_sensor,
    )
    runtime = FixedLagContinuityRuntime(line_assets)
    feature_rows = [
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    ]

    committed = []
    for idx, row in enumerate(feature_rows):
        committed.extend(runtime.step(row, sample_timestamp_ns=idx, finalize_timestamp_ns=idx))
    committed.extend(runtime.flush(finalize_timestamp_ns=10))

    assert [row.loop_node_index for row in committed] == [1, 0, 0]
