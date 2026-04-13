from pathlib import Path

import pandas as pd

from acies.controller.ict_ground_truth import SensorPoint, project_sensor_positions


def _make_sensor_geometry() -> pd.DataFrame:
    points = [
        SensorPoint("rs8", 40.0000, -88.0000),
        SensorPoint("rs10", 40.0000, -88.00002),
        SensorPoint("rs6", 40.0004, -88.0000),
        SensorPoint("rs7", 40.0004, -88.00002),
        SensorPoint("rs3", 40.0008, -88.0000),
        SensorPoint("rs5", 40.0008, -88.00002),
        SensorPoint("rs1", 40.0012, -88.0000),
        SensorPoint("rs2", 40.0012, -88.00002),
    ]
    return pd.DataFrame(project_sensor_positions(points, pair_gap_threshold_m=10.0))


def test_station_geometry_ordering_is_monotonic():
    df = _make_sensor_geometry()
    stations = df.groupby("station_id", as_index=False)["station_axis_m"].first().sort_values("station_id")
    values = stations["station_axis_m"].tolist()
    assert values == sorted(values)


def test_sensor_pairs_share_station_id():
    df = _make_sensor_geometry()
    stations = df.set_index("node")["station_id"].to_dict()
    assert stations["rs8"] == stations["rs10"]
    assert stations["rs6"] == stations["rs7"]
    assert stations["rs3"] == stations["rs5"]
    assert stations["rs1"] == stations["rs2"]
