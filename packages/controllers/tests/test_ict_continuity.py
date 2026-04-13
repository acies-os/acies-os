from __future__ import annotations

import math
import sys
from pathlib import Path

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
