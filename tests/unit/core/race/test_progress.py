"""Tests for mlracecar.core.race.progress: locating cars along the centerline."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlracecar.core.geometry import wrap_angle
from mlracecar.core.race.progress import RoadLocator
from mlracecar.core.track.model import Track

ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 12)
LOCATOR = RoadLocator(CIRCLE, reach=5.0)


def around(distance: float) -> float:
    """``distance`` folded into ``(-half a lap, half a lap]``."""
    half = CIRCLE.length / 2
    return float(np.mod(distance + half, 2 * half) - half)


@given(st.floats(0.0, CIRCLE.length, exclude_max=True), st.floats(-20.0, 20.0))
def test_points_are_found_where_they_are_along_the_lap(arc_length: float, offset: float) -> None:
    point = CIRCLE.pose_at(arc_length, offset).position.reshape(1, 2)

    found = LOCATOR.locate(point)

    # The centerline is a chain of 0.5 m straight pieces. Off the middle of a bend, the square
    # from a straight piece meets it up to offset · curvature · 0.5 m / 2 away from where the
    # square from the curve would: 1 cm at 3 m off the middle of this 60 m radius circle.
    along_error = 0.01 + abs(offset) / 60 * 0.5 / 2
    assert around(found.arc_length[0] - arc_length) == pytest.approx(0.0, abs=along_error)
    assert found.offset[0] == pytest.approx(offset, abs=0.01)
    assert wrap_angle(found.heading[0] - CIRCLE.pose_at(arc_length).heading) == pytest.approx(
        0.0, abs=0.01
    )
    assert 0.0 <= found.arc_length[0] < CIRCLE.length


def test_searching_near_the_last_spot_finds_the_same_as_searching_everywhere() -> None:
    arc_length = np.linspace(0, CIRCLE.length, 200, endpoint=False)
    before = LOCATOR.locate(CIRCLE.pose_at(arc_length, 2.0).position)
    moved = CIRCLE.pose_at(arc_length + 4.0, -1.0).position

    near, everywhere = LOCATOR.locate(moved, near=before.segment), LOCATOR.locate(moved)

    for got, expected in zip(near, everywhere, strict=True):
        np.testing.assert_array_equal(got, expected)


def test_a_car_stays_on_its_own_stretch_of_road() -> None:
    # A car last seen 100 m further on, that suddenly shows up here (as if it cut across the
    # infield), is found on its own stretch, not here.
    point = CIRCLE.pose_at([50.0]).position
    last_seen = LOCATOR.locate(CIRCLE.pose_at([150.0]).position).segment

    found = LOCATOR.locate(point, near=last_seen)

    assert abs(around(found.arc_length[0] - 150.0)) < 20.0
    assert abs(found.offset[0]) > 20.0


def test_many_cars_are_found_in_chunks_like_one_at_a_time() -> None:
    points = CIRCLE.pose_at(np.linspace(0, 300, 150), 3.0).position

    together = LOCATOR.locate(points)
    alone = [LOCATOR.locate(point.reshape(1, 2)) for point in points]

    for index, got in enumerate(together):
        np.testing.assert_array_equal(got, np.concatenate([single[index] for single in alone]))
