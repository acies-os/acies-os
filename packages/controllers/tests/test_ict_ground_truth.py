from pathlib import Path

from acies.controller.ict_ground_truth import SensorPoint, parse_gps_run_file, project_sensor_positions


def test_parse_gps_run_file():
    parsed = parse_gps_run_file(Path("run7_miata_gps.parquet"))
    assert parsed is not None
    assert parsed.run_id == 7
    assert parsed.label == "miata"

    parsed = parse_gps_run_file(Path("run3_gps.parquet"))
    assert parsed is not None
    assert parsed.run_id == 3
    assert parsed.label is None

    assert parse_gps_run_file(Path("run3_rs1_mic.parquet")) is None


def test_project_sensor_positions_groups_pairs_into_stations():
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

    rows = project_sensor_positions(points, pair_gap_threshold_m=10.0)
    order = [row["node"] for row in rows]
    assert order in (["rs8", "rs10", "rs6", "rs7", "rs3", "rs5", "rs1", "rs2"], ["rs1", "rs2", "rs3", "rs5", "rs6", "rs7", "rs8", "rs10"])

    stations = {row["node"]: row["station_id"] for row in rows}
    assert stations["rs8"] == stations["rs10"]
    assert stations["rs6"] == stations["rs7"]
    assert stations["rs3"] == stations["rs5"]
    assert stations["rs1"] == stations["rs2"]
    assert len(set(stations.values())) == 4
