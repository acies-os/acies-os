from __future__ import annotations

import math
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from acies.controller.ict_continuity import (
    ContinuityLatticeConfig,
    build_point_lattice,
    infer_loop_sensor_order,
    signed_loop_delta,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_ict_continuity_layer.py"
SPEC = importlib.util.spec_from_file_location("eval_ict_continuity_layer_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EVAL_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL_SCRIPT)


def _sensor_geometry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"node": "n1", "station_id": 1, "cross_m": -10.0, "ref_latitude": 0.0, "ref_longitude": 0.0},
            {"node": "p1", "station_id": 1, "cross_m": 10.0, "ref_latitude": 0.0, "ref_longitude": 0.0},
            {"node": "n2", "station_id": 2, "cross_m": -10.0, "ref_latitude": 0.0, "ref_longitude": 0.0},
            {"node": "p2", "station_id": 2, "cross_m": 10.0, "ref_latitude": 0.0, "ref_longitude": 0.0},
        ]
    )


def _samples() -> pd.DataFrame:
    rows = []
    for run_id, offset in [(0, 0.0), (1, 0.5)]:
        for idx, node in enumerate(["n1", "n2", "p2", "p1"]):
            base_y = float(idx * 20.0)
            for step in range(12):
                rows.append(
                    {
                        "run_id": run_id,
                        "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=len(rows)),
                        "nearest_sensor": node,
                        "nearest_sensor_distance_m": 3.0 + 0.5 * abs(step - 6),
                        "latitude": (base_y + step + offset) / 111_000.0,
                        "longitude": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _hybrid_samples() -> pd.DataFrame:
    rows = []
    feature_cols = ["n1__mic", "n2__mic", "p2__mic", "p1__mic"]
    timestamp = pd.Timestamp("2024-01-01 00:00:00")
    for run_id, offset in [(0, 0.0), (1, 0.6)]:
        for node_idx in range(20):
            for rep in range(3):
                base = np.zeros(len(feature_cols), dtype=float)
                base[node_idx % len(feature_cols)] = 1.0 + 0.1 * rep + offset
                rows.append(
                    {
                        "run_id": run_id,
                        "label": "demo",
                        "timestamp": timestamp,
                        "nearest_sensor": ["n1", "n2", "p2", "p1"][node_idx // 5],
                        "nearest_sensor_distance_m": 2.0 + rep,
                        "latitude": (node_idx * 2.0 + rep) / 111_000.0,
                        "longitude": 0.0,
                        "pred_station": 1 + (node_idx // 5) % 2,
                        "pred_side": "negative_cross" if (node_idx // 5) < 2 else "positive_cross",
                        "pred_direction": "toward_S4",
                        "pred_station_margin": 1.0,
                        "gt_loop_node_index": node_idx,
                        "n1__mic": base[0],
                        "n2__mic": base[1],
                        "p2__mic": base[2],
                        "p1__mic": base[3],
                    }
                )
                timestamp += pd.Timedelta(seconds=1)
    return pd.DataFrame(rows)


def test_infer_loop_sensor_order() -> None:
    order = infer_loop_sensor_order(_sensor_geometry())
    assert order == ["n1", "n2", "p2", "p1"]


def test_signed_loop_delta_wraps_short_way() -> None:
    assert signed_loop_delta(39, 1, n_nodes=40) == 2
    assert signed_loop_delta(1, 39, n_nodes=40) == -2
    assert signed_loop_delta(5, 8, n_nodes=40) == 3


def test_build_point_lattice_emits_five_points_per_sensor() -> None:
    lattice = build_point_lattice(_samples(), _sensor_geometry(), config=ContinuityLatticeConfig(point_count=5))
    counts = lattice.groupby("sensor_node").size().to_dict()
    assert counts == {"n1": 5, "n2": 5, "p1": 5, "p2": 5}
    assert lattice["loop_node_index"].tolist() == list(range(20))
    for _node, group in lattice.groupby("sensor_node"):
        assert group["point_index"].tolist() == [1, 2, 3, 4, 5]
        assert group["u_m"].is_monotonic_increasing


def test_annotate_eval_loops_marks_only_manifest_rows() -> None:
    hybrid = _hybrid_samples()[["run_id", "label", "timestamp"]].copy()
    manifest = pd.DataFrame(
        [
            {
                "run_id": 0,
                "label": "demo",
                "loop_id": 7,
                "loop_rank_in_run": 0,
                "start_timestamp": hybrid.iloc[2]["timestamp"],
                "end_timestamp": hybrid.iloc[5]["timestamp"],
                "start_elapsed_s": 2.0,
                "end_elapsed_s": 5.0,
                "n_points": 4,
                "fold_id": 1,
            }
        ]
    )
    annotated = EVAL_SCRIPT.annotate_eval_loops(hybrid, manifest)
    assert int((annotated["eval_split"] == "test").sum()) == 4
    assert annotated.loc[2:5, "loop_id"].tolist() == [7, 7, 7, 7]
    assert annotated.loc[2:5, "eval_split"].tolist() == ["test"] * 4
    assert set(annotated.drop(index=range(2, 6))["eval_split"]) == {"train"}


def test_build_observation_models_return_expected_shapes() -> None:
    hybrid = _hybrid_samples()
    lattice = pd.DataFrame({"loop_node_index": list(range(20))})
    mean_model = EVAL_SCRIPT._build_observation_model(hybrid, lattice, ["n1__mic", "n2__mic", "p2__mic", "p1__mic"], "pooled_mean")
    median_model = EVAL_SCRIPT._build_observation_model(hybrid, lattice, ["n1__mic", "n2__mic", "p2__mic", "p1__mic"], "pooled_median")
    var_model = EVAL_SCRIPT._build_observation_model(hybrid, lattice, ["n1__mic", "n2__mic", "p2__mic", "p1__mic"], "diagvar_mean")
    kmeans_model = EVAL_SCRIPT._build_observation_model(hybrid, lattice, ["n1__mic", "n2__mic", "p2__mic", "p1__mic"], "kmeans2_min")

    assert mean_model["templates"].shape == (20, 4)
    assert median_model["templates"].shape == (20, 4)
    assert var_model["variances"].shape == (20, 4)
    assert np.all(var_model["variances"] > 0.0)
    assert kmeans_model["centers"].shape == (20, 2, 4)


def test_kmeans_model_falls_back_on_low_sample_nodes() -> None:
    hybrid = _hybrid_samples()
    sparse = hybrid[hybrid["gt_loop_node_index"] != 0].copy()
    sparse = pd.concat([sparse, hybrid[hybrid["gt_loop_node_index"] == 0].head(2)], ignore_index=True)
    lattice = pd.DataFrame({"loop_node_index": list(range(20))})
    model = EVAL_SCRIPT._build_observation_model(sparse, lattice, ["n1__mic", "n2__mic", "p2__mic", "p1__mic"], "kmeans2_min")
    assert np.allclose(model["centers"][0, 0], model["centers"][0, 1])
