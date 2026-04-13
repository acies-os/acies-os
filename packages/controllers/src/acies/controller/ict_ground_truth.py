from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

RUN_GPS_PATTERN = re.compile(r"run(?P<run>\d+)(?:_(?P<label>[A-Za-z0-9\-]+))?_gps\.parquet$")

_DEG_TO_RAD = math.pi / 180.0
_EARTH_R = 6_371_000.0


@dataclass(frozen=True)
class GpsRunFile:
    run_id: int
    label: str | None
    path: Path


@dataclass(frozen=True)
class SensorPoint:
    node: str
    latitude: float
    longitude: float


def parse_gps_run_file(path: Path) -> GpsRunFile | None:
    match = RUN_GPS_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    label = match.group("label")
    return GpsRunFile(run_id=int(match.group("run")), label=label, path=path)


def latlon_to_xy_m(lat: float, lon: float, ref_lat: float, ref_lon: float) -> tuple[float, float]:
    cos_ref = math.cos(ref_lat * _DEG_TO_RAD)
    x = (lon - ref_lon) * _DEG_TO_RAD * _EARTH_R * cos_ref
    y = (lat - ref_lat) * _DEG_TO_RAD * _EARTH_R
    return x, y


def fit_sensor_axis(points: list[SensorPoint]) -> tuple[tuple[float, float], tuple[float, float], list[tuple[float, float]]]:
    if len(points) < 2:
        raise ValueError("Need at least two sensor points to fit an axis")

    ref_lat = sum(point.latitude for point in points) / len(points)
    ref_lon = sum(point.longitude for point in points) / len(points)
    xy_points = [latlon_to_xy_m(point.latitude, point.longitude, ref_lat, ref_lon) for point in points]

    mean_x = sum(x for x, _ in xy_points) / len(xy_points)
    mean_y = sum(y for _, y in xy_points) / len(xy_points)
    centered = [(x - mean_x, y - mean_y) for x, y in xy_points]

    sxx = sum(x * x for x, _ in centered)
    syy = sum(y * y for _, y in centered)
    sxy = sum(x * y for x, y in centered)

    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    axis = (math.cos(angle), math.sin(angle))
    if axis[1] > 0 or (abs(axis[1]) < 1e-9 and axis[0] < 0):
        axis = (-axis[0], -axis[1])

    return (ref_lat, ref_lon), axis, xy_points


def project_sensor_positions(
    points: list[SensorPoint], pair_gap_threshold_m: float = 10.0
) -> list[dict[str, float | int | str]]:
    ref_latlon, axis, xy_points = fit_sensor_axis(points)
    axis_x, axis_y = axis
    center_x = sum(x for x, _ in xy_points) / len(xy_points)
    center_y = sum(y for _, y in xy_points) / len(xy_points)
    normal_x, normal_y = -axis_y, axis_x

    rows: list[dict[str, float | int | str]] = []
    for point, (x, y) in zip(points, xy_points):
        dx = x - center_x
        dy = y - center_y
        axis_m = dx * axis_x + dy * axis_y
        cross_m = dx * normal_x + dy * normal_y
        rows.append(
            {
                "node": point.node,
                "latitude": point.latitude,
                "longitude": point.longitude,
                "x_m": x,
                "y_m": y,
                "axis_m": axis_m,
                "cross_m": cross_m,
                "ref_latitude": ref_latlon[0],
                "ref_longitude": ref_latlon[1],
            }
        )

    rows.sort(key=lambda row: float(row["axis_m"]))
    station_id = 0
    prev_axis = None
    for row in rows:
        axis_m = float(row["axis_m"])
        if prev_axis is None or axis_m - prev_axis > pair_gap_threshold_m:
            station_id += 1
        row["station_id"] = station_id
        prev_axis = axis_m

    station_means: dict[int, tuple[float, float, float, float]] = {}
    for current_station in sorted({int(row["station_id"]) for row in rows}):
        station_rows = [row for row in rows if int(row["station_id"]) == current_station]
        station_means[current_station] = (
            sum(float(row["axis_m"]) for row in station_rows) / len(station_rows),
            sum(float(row["cross_m"]) for row in station_rows) / len(station_rows),
            sum(float(row["x_m"]) for row in station_rows) / len(station_rows),
            sum(float(row["y_m"]) for row in station_rows) / len(station_rows),
        )

    for row in rows:
        station_axis_m, station_cross_m, station_x_m, station_y_m = station_means[int(row["station_id"])]
        row["station_axis_m"] = station_axis_m
        row["station_cross_m"] = station_cross_m
        row["station_x_m"] = station_x_m
        row["station_y_m"] = station_y_m

    return rows
