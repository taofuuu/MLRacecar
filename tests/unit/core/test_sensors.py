"""Tests for mlracecar.core.sensors: distance rays that stop at the road's edges."""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import mlracecar.core.sensors as sensors
from circuits import gp_circuit
from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.geometry import FloatArray, cast_rays, cross, unit_vector
from mlracecar.core.race.rules import RaceRules
from mlracecar.core.sensors import RaySensor, RaySettings
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.core.world import StartPosition, World
from mlracecar.io.track_file import read_track_file

CAR = VehicleConfig().to_params()
TIMING = SimulationConfig().to_timing()
TRACKS_DIR = Path(__file__).parents[3] / "tracks"


def circle(radius: float, width: float = 12.0, points: int = 48) -> Track:
    """A round track, driven counter-clockwise: its left edge is the inside."""
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)
    return Track.build(radius * np.column_stack([np.cos(angles), np.sin(angles)]), [width] * points)


CIRCLE = circle(60.0)
"""Edges at 54 m (left, inside) and 66 m (right, outside) from the centre."""


def cars_at(track: Track, positions: list[tuple[float, float]], yaws: list[float]) -> Snapshot:
    """Cars standing at these places, each with its place along the lap worked out."""
    cars = VehicleState.at_rest(np.array(positions, dtype=float), np.array(yaws, dtype=float))
    return Snapshot(0, 0.0, cars, RaceRules(track, reach=1.0).start(cars), ())


def rays(count: int, field_of_view: float = math.pi, max_range: float = 100.0) -> RaySettings:
    return RaySettings(count=count, field_of_view=field_of_view, max_range=max_range)


def circle_hits(origins: FloatArray, directions: FloatArray, radius: float) -> FloatArray:
    """How far each ray goes before it meets a circle round (0, 0); inf if it never does."""
    along = np.einsum("...i,...i->...", origins, directions)
    gap = along**2 - (np.einsum("...i,...i->...", origins, origins) - radius**2)
    root = np.sqrt(np.maximum(gap, 0.0))
    nearest = np.where(-along - root >= 0, -along - root, -along + root)
    return np.where((gap >= 0) & (nearest >= 0), nearest, np.inf)


def everywhere(track: Track, snapshot: Snapshot, sensor: RaySensor) -> FloatArray:
    """The readings from testing every ray against every piece of both edges."""
    cars = snapshot.cars
    directions = unit_vector(cars.yaw[:, None] + sensor.angles).reshape(-1, 2)
    origins = np.repeat(cars.position, len(sensor.angles), axis=0)
    nearest = [
        cast_rays(origins, directions, edge, np.roll(edge, -1, axis=0), sensor.settings.max_range)
        for edge in (track.left, track.right)
    ]
    return np.minimum(*nearest).reshape(len(cars), -1)


# --------------------------------------------------------------------------- #
# Where the rays point
# --------------------------------------------------------------------------- #


def test_rays_fan_evenly_across_the_field_of_view_from_right_to_left() -> None:
    sensor = RaySensor(CIRCLE, rays(5))

    np.testing.assert_allclose(sensor.angles, [-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])


def test_a_single_ray_points_straight_ahead() -> None:
    assert RaySensor(CIRCLE, rays(1)).angles.tolist() == [0.0]


def test_rays_all_the_way_round_do_not_repeat_the_first() -> None:
    sensor = RaySensor(CIRCLE, rays(4, field_of_view=2 * math.pi))

    np.testing.assert_allclose(sensor.angles, [-np.pi, -np.pi / 2, 0, np.pi / 2])


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (rays(0), "at least 1 ray"),
        (rays(5, field_of_view=0.0), "field of view"),
        (rays(5, field_of_view=7.0), "field of view"),
        (rays(5, max_range=0.0), "range"),
    ],
)
def test_impossible_settings_are_refused(settings: RaySettings, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RaySensor(CIRCLE, settings)


# --------------------------------------------------------------------------- #
# What they measure, on a round track where the answers are known
# --------------------------------------------------------------------------- #


def test_a_car_in_the_middle_sees_both_edges_and_the_bend_ahead() -> None:
    # At (60, 0), driving up the page: right is the outside edge, left the inside one, and
    # straight ahead the outside edge curves across the car's path.
    readings = RaySensor(CIRCLE, rays(3)).sense(cars_at(CIRCLE, [(60, 0)], [np.pi / 2]))

    np.testing.assert_allclose(readings.distance, [[6.0, math.sqrt(66**2 - 60**2), 6.0]], atol=0.01)


def test_a_car_near_the_outside_edge_sees_it_closer() -> None:
    readings = RaySensor(CIRCLE, rays(3)).sense(cars_at(CIRCLE, [(63, 0)], [np.pi / 2]))

    np.testing.assert_allclose(readings.distance, [[3.0, math.sqrt(66**2 - 63**2), 9.0]], atol=0.01)


def test_a_car_facing_across_the_road_sees_the_edge_in_front_of_it() -> None:
    readings = RaySensor(CIRCLE, rays(3)).sense(cars_at(CIRCLE, [(60, 0)], [0.0]))

    ahead = math.sqrt(66**2 - 60**2)  # sideways along the road, both ways, to the outside edge
    np.testing.assert_allclose(readings.distance, [[ahead, 6.0, ahead]], atol=0.01)


@given(
    angle=st.floats(-np.pi, np.pi),
    offset=st.floats(-5.5, 5.5),
    yaw=st.floats(-np.pi, np.pi),
)
def test_every_ray_stops_at_the_nearer_of_the_two_circles(
    angle: float, offset: float, yaw: float
) -> None:
    position = (60 + offset) * np.array([math.cos(angle), math.sin(angle)])
    sensor = RaySensor(CIRCLE, rays(19, field_of_view=2 * math.pi))

    readings = sensor.sense(cars_at(CIRCLE, [tuple(position)], [yaw]))

    directions = unit_vector(yaw + sensor.angles)
    edges = np.minimum(
        circle_hits(position, directions, 54.0), circle_hits(position, directions, 66.0)
    )
    # Rays that only just graze the inside edge are left out: there, the tiny difference between
    # the track's smooth curve and a true circle moves the meeting point a long way.
    grazing = np.abs(np.abs(cross(position, directions)) - 54.0) < 0.5
    np.testing.assert_allclose(
        readings.distance[0][~grazing], np.minimum(edges, 100.0)[~grazing], atol=0.01
    )


def test_rays_that_meet_no_edge_within_range_read_the_range() -> None:
    wide_bend = circle(2000.0, points=200)  # the outside edge is 155 m ahead
    snapshot = cars_at(wide_bend, [(2000, 0)], [np.pi / 2])

    readings = RaySensor(wide_bend, rays(3)).sense(snapshot)

    np.testing.assert_allclose(readings.distance, [[6.0, 100.0, 6.0]], atol=0.01)
    np.testing.assert_allclose(readings.normalized, [[0.06, 1.0, 0.06]], atol=1e-4)


def test_normalized_readings_are_fractions_of_the_range() -> None:
    readings = RaySensor(CIRCLE, rays(15, max_range=20.0)).sense(
        cars_at(CIRCLE, [(60, 0), (0, 58)], [np.pi / 2, np.pi])
    )

    np.testing.assert_allclose(readings.normalized, readings.distance / 20.0)
    assert readings.distance.max() == 20.0
    assert ((readings.normalized >= 0) & (readings.normalized <= 1)).all()


def test_each_ray_ends_where_it_meets_the_edge() -> None:
    snapshot = cars_at(CIRCLE, [(60, 0)], [np.pi / 2])
    sensor = RaySensor(CIRCLE, rays(15))

    readings = sensor.sense(snapshot)

    expected = snapshot.cars.position[:, None] + readings.distance[..., None] * unit_vector(
        np.pi / 2 + sensor.angles
    )
    np.testing.assert_allclose(readings.end, expected)
    distance_from_centre = np.hypot(readings.end[..., 0], readings.end[..., 1])
    assert np.all(
        np.isclose(distance_from_centre, 54, atol=0.01)
        | np.isclose(distance_from_centre, 66, atol=0.01)
    )


# --------------------------------------------------------------------------- #
# Many cars, and the shortcuts that keep it fast
# --------------------------------------------------------------------------- #


def test_each_car_reads_the_same_alone_as_among_others() -> None:
    places = [(60.0, 0.0), (0.0, -63.0), (-42.0, 42.0), (57.0, 3.0)]
    headings = [np.pi / 2, 0.3, -2.0, np.pi]
    sensor = RaySensor(CIRCLE, rays(15))

    together = sensor.sense(cars_at(CIRCLE, places, headings)).distance

    alone = [
        sensor.sense(cars_at(CIRCLE, [place], [yaw])).distance[0]
        for place, yaw in zip(places, headings, strict=True)
    ]
    np.testing.assert_array_equal(together, alone)


def test_many_cars_are_worked_out_a_few_at_a_time_with_the_same_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = World(
        CIRCLE,
        KinematicBicycle(CAR),
        TIMING,
        10,
        np.random.default_rng(0),
        start=StartPosition.RANDOM,
    ).snapshot
    sensor = RaySensor(CIRCLE, rays(15))
    at_once = sensor.sense(snapshot).distance

    monkeypatch.setattr(sensors, "_CHUNK_SIZE", 1)  # one car at a time

    np.testing.assert_array_equal(sensor.sense(snapshot).distance, at_once)


GP = gp_circuit()
TRACKS = {
    "gp": Track.build(GP.points, GP.widths),
    "technical": read_track_file(TRACKS_DIR / "technical.json").to_track(),
    "oval": read_track_file(TRACKS_DIR / "oval.json").to_track(),
    "short": circle(12.0, width=8.0),  # a lap shorter than the rays' reach
}
SENSORS = {
    name: RaySensor(track, rays(13, field_of_view=2 * math.pi)) for name, track in TRACKS.items()
}


@settings(max_examples=40)
@given(name=st.sampled_from(sorted(TRACKS)), seed=st.integers(0, 2**32 - 1))
def test_the_shortcuts_give_the_same_readings_as_testing_every_piece_of_edge(
    name: str, seed: int
) -> None:
    # Cars anywhere on the road, facing any way: the readings are exact, not approximate.
    track, sensor = TRACKS[name], SENSORS[name]
    rng = np.random.default_rng(seed)
    world = World(track, KinematicBicycle(CAR), TIMING, 6, rng, start=StartPosition.RANDOM)
    snapshot = world.snapshot
    snapshot = replace(snapshot, cars=replace(snapshot.cars, yaw=rng.uniform(-np.pi, np.pi, 6)))

    readings = sensor.sense(snapshot)

    np.testing.assert_allclose(readings.distance, everywhere(track, snapshot, sensor), atol=1e-9)
