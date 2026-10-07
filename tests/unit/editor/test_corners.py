"""Tests for mlracecar.editor.corners: rounding a sharp point into a bend of a chosen radius."""

import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlracecar.core.geometry import FloatArray
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import has_errors, validate
from mlracecar.editor.corners import (
    SHARP_TURN,
    CornerError,
    RoundedCorner,
    corner_limits,
    round_corner,
)
from strategies import track_like_points

SQUARE = np.array([(100.0, -100.0), (100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0)])
WIDTHS = np.full(4, 12.0)
SKETCH = np.array(
    [(0, 0), (300, -20), (420, 80), (380, 260), (200, 220), (120, 330), (-80, 300), (-60, 140)],
    dtype=float,
)


def corner_scene(turn_degrees: float, arm: float = 200.0) -> FloatArray:
    """A loop whose point 1, at the origin, turns by ``turn_degrees`` (positive turns left).

    The road arrives along +x from point 0 and leaves towards point 2, ``arm`` metres each way;
    points 3 and 4 close the loop on the inside of the turn, well away from the corner.
    """
    turn = math.radians(turn_degrees)
    side = math.copysign(1.0, turn_degrees)
    a, p = np.array([-arm, 0.0]), np.array([0.0, 0.0])
    b = arm * np.array([math.cos(turn), math.sin(turn)])
    c = b + arm * np.array([math.cos(turn + side * 1.2), math.sin(turn + side * 1.2)])
    d = a + np.array([-arm / 2, side * 1.5 * arm])
    return np.array([a, p, b, c, d])


def tightest_on_bend(points: FloatArray, index: int, rounded: RoundedCorner) -> float:
    """The tightest radius of the actual track along the new bend.

    The bend is where the new points leave the two straights; the track is measured within
    that distance of the old corner, which stays clear of the neighbouring corners.
    """
    corner = points[index]
    incoming = corner - points[index - 1]
    outgoing = points[(index + 1) % len(points)] - corner
    new = [p for p in rounded.points if not np.any(np.all(points == p, axis=1))]
    off = [
        p
        for p in new
        if min(_distance_to_line(p, corner, incoming), _distance_to_line(p, corner, outgoing))
        > 1e-6
    ]
    reach = max(float(np.linalg.norm(p - corner)) for p in off)
    track = Track.build(rounded.points, rounded.widths)
    near = np.linalg.norm(track.centerline.points - corner, axis=1) <= reach
    return float(1 / np.abs(track.centerline.curvature[near]).max())


def _distance_to_line(point: FloatArray, origin: FloatArray, direction: FloatArray) -> float:
    unit = direction / np.linalg.norm(direction)
    offset = point - origin
    return abs(float(unit[0] * offset[1] - unit[1] * offset[0]))


def radii_across(points: FloatArray, widths: FloatArray, index: int) -> list[float]:
    limits = corner_limits(points, widths, index)
    return [float(r) for r in np.linspace(limits.smallest, limits.largest, 4)]


# --------------------------------------------------------------------------- #
# The acceptance criteria
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("share", [0.0, 1 / 3, 2 / 3, 1.0])
def test_a_90_degree_corner_comes_out_within_10_percent_of_the_radius(share: float) -> None:
    limits = corner_limits(SQUARE, WIDTHS, 1)
    radius = limits.smallest + share * (limits.largest - limits.smallest)
    rounded = round_corner(SQUARE, WIDTHS, 1, radius)
    assert tightest_on_bend(SQUARE, 1, rounded) == pytest.approx(radius, rel=0.1)


@pytest.mark.parametrize("turn", [20, -20, 165, -165, 10, 45, -90, 135])
def test_kinks_and_hairpins_come_out_within_10_percent_both_ways(turn: float) -> None:
    points = corner_scene(turn)
    widths = np.full(len(points), 12.0)
    for radius in radii_across(points, widths, 1):
        rounded = round_corner(points, widths, 1, radius)
        assert tightest_on_bend(points, 1, rounded) == pytest.approx(radius, rel=0.1)


def test_a_rough_sketch_rounded_at_every_corner_has_no_track_problems() -> None:
    points, widths = SKETCH, np.full(len(SKETCH), 12.0)
    for corner in SKETCH:
        index = int(np.flatnonzero(np.all(points == corner, axis=1))[0])
        radius = min(30.0, corner_limits(points, widths, index).largest)
        rounded = round_corner(points, widths, index, radius)
        points, widths = rounded.points, rounded.widths
    assert validate(points, widths) == []


def turn_at(points: FloatArray, index: int) -> float:
    """How far the road turns at a point, in radians."""
    incoming = points[index] - points[index - 1]
    outgoing = points[(index + 1) % len(points)] - points[index]
    cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
    return abs(math.atan2(cross, float(incoming @ outgoing)))


@settings(max_examples=30, deadline=None)
@given(track_like_points(clockwise=None), st.floats(8.0, 40.0))
def test_any_sketch_rounded_at_every_corner_can_be_raced(sketch: FloatArray, radius: float) -> None:
    # Every point that can be rounded is, as near the radius as fits. Points where the road
    # barely turns, and gentle ones without room for their wide bend, stay as they are. A sharp
    # corner with no room left (its neighbours took it) can't be raced, so that sketch is skipped.
    points, widths = sketch, np.full(len(sketch), 12.0)
    for corner in sketch:
        index = int(np.flatnonzero(np.all(points == corner, axis=1))[0])
        try:
            limits = corner_limits(points, widths, index)
        except CornerError:
            assume(turn_at(points, index) < SHARP_TURN)
            continue
        rounded = round_corner(
            points, widths, index, float(np.clip(radius, limits.smallest, limits.largest))
        )
        points, widths = rounded.points, rounded.widths
    assert not has_errors(validate(points, widths))


# --------------------------------------------------------------------------- #
# The shape
# --------------------------------------------------------------------------- #


def test_a_right_turn_is_the_mirror_image_of_a_left_turn() -> None:
    left, right = corner_scene(60), corner_scene(-60)
    widths = np.full(5, 12.0)
    rounded_left = round_corner(left, widths, 1, 30.0)
    rounded_right = round_corner(right, widths, 1, 30.0)
    np.testing.assert_allclose(rounded_right.points * [1, -1], rounded_left.points, atol=1e-9)


def test_the_neighbouring_points_stay_and_point_0_stays_first() -> None:
    rounded = round_corner(SQUARE, WIDTHS, 1, 30.0)
    assert tuple(rounded.points[0]) == tuple(SQUARE[0])
    for neighbour in (SQUARE[2], SQUARE[3]):
        assert np.any(np.all(rounded.points == neighbour, axis=1))
    assert not np.any(np.all(rounded.points == SQUARE[1], axis=1))  # the sharp point is gone


def test_rounding_point_0_puts_the_middle_of_the_bend_on_the_start_line() -> None:
    rounded = round_corner(SQUARE, WIDTHS, 0, 30.0)
    assert rounded.middle == 0
    x, y = rounded.points[0]
    # The corner at (100, -100) turns 90 degrees, so its bend's middle is on the diagonal x = -y.
    assert x + y == pytest.approx(0.0, abs=1e-6)
    assert x < 100


def test_the_middle_is_the_bend_point_nearest_the_old_corner() -> None:
    rounded = round_corner(SQUARE, WIDTHS, 2, 30.0)
    distances = np.linalg.norm(rounded.points - SQUARE[2], axis=1)
    assert rounded.middle == int(np.argmin(distances))


def test_the_bend_keeps_the_corner_width_and_the_straights_blend_to_the_neighbours() -> None:
    widths = np.array([10.0, 14.0, 20.0, 10.0])
    rounded = round_corner(SQUARE, widths, 1, 30.0)
    assert rounded.widths[rounded.middle] == 14.0
    assert np.all((rounded.widths >= 10.0) & (rounded.widths <= 20.0))


def test_points_on_the_straights_are_replaced_so_order_does_not_matter() -> None:
    before = corner_limits(SQUARE, WIDTHS, 2)
    rounded = round_corner(SQUARE, WIDTHS, 1, before.largest)
    index = int(np.flatnonzero(np.all(rounded.points == SQUARE[2], axis=1))[0])
    after = corner_limits(rounded.points, rounded.widths, index)
    assert after.largest == pytest.approx(before.largest)


def test_points_on_the_straights_keep_their_widths() -> None:
    # A narrower point halfway along the straight after the corner.
    points = np.insert(SQUARE, 2, [(0.0, 100.0)], axis=0)
    widths = np.array([12.0, 12.0, 8.0, 12.0, 12.0])
    rounded = round_corner(points, widths, 1, 20.0)
    on_straight = np.isclose(rounded.points[:, 1], 100.0) & (rounded.points[:, 0] < 60)
    assert rounded.widths[on_straight].min() < 11.5  # still narrower towards the middle
    assert rounded.widths[on_straight].min() >= 8.0


# --------------------------------------------------------------------------- #
# Limits and refusals
# --------------------------------------------------------------------------- #


def test_the_smallest_radius_keeps_the_inside_edge_clear() -> None:
    assert corner_limits(SQUARE, WIDTHS, 1).smallest == 8.0  # 1.05 * (12 / 2 + 1), rounded up
    assert corner_limits(SQUARE, np.full(4, 20.0), 1).smallest == 12.0
    assert corner_limits(SQUARE, np.full(4, 6.0), 1).smallest == 7.0  # 1.05 * the 6 m minimum


def test_the_largest_radius_leaves_half_of_each_straight_for_the_next_corner() -> None:
    limits = corner_limits(SQUARE, WIDTHS, 1)
    rounded = round_corner(SQUARE, WIDTHS, 1, limits.largest)
    # The corner at (100, 100) joins the straights x = 100 and y = 100; the bend leaves them
    # no earlier than their halfway points, (100, 0) and (0, 100).
    bend = [p for p in rounded.points if abs(p[0] - 100) > 0.01 and abs(p[1] - 100) > 0.01]
    bend = [p for p in bend if p[0] > -50 and p[1] > -50]  # not the far corner
    assert min(min(x, y) for x, y in bend) >= -1e-6


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (np.array([(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (50.0, 80.0)]), "barely turns"),
        (np.array([(0.0, 0.0), (50.0, 0.0), (100.0, 4.0), (50.0, 80.0)]), "barely turns"),
        (np.array([(0.0, 0.0), (100.0, 0.0), (0.0, 0.0001), (-50, 80)]), "doubles straight back"),
        (np.array([(0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (50, 80)]), "on top of"),
        (np.array([(0.0, 0.0), (10.0, 0.0)]), "at least 3 points"),
        (np.array([(-10.0, 0.0), (0.0, 0.0), (0.0, 10.0), (-10, 10)]), "no radius fits"),
    ],
)
def test_points_that_cannot_be_rounded_say_why(points: FloatArray, message: str) -> None:
    with pytest.raises(CornerError, match=message):
        corner_limits(points, np.full(len(points), 12.0), 1)


def test_a_radius_that_does_not_fit_says_which_do() -> None:
    with pytest.raises(CornerError, match=r"choose between 8 m and \d+\.\d m"):
        round_corner(SQUARE, WIDTHS, 1, 200.0)
    with pytest.raises(CornerError, match="doesn't fit"):
        round_corner(SQUARE, WIDTHS, 1, 5.0)


def test_after_saving_and_reopening_points_on_the_straights_are_still_replaced() -> None:
    # Saved files keep positions to the millimetre (`TrackDocument`); the straights' points
    # must still count as on them, or the next corner would get less room.
    rounded = round_corner(SQUARE, WIDTHS, 1, 40.0)
    index = int(np.flatnonzero(np.all(rounded.points == SQUARE[2], axis=1))[0])
    exact = corner_limits(rounded.points, rounded.widths, index)
    reopened = corner_limits(np.round(rounded.points, 3), rounded.widths, index)
    assert reopened.largest == pytest.approx(exact.largest, rel=1e-3)


def test_a_duplicate_point_beyond_a_straight_does_not_get_in_the_way() -> None:
    # Two points on top of each other are a track problem of their own; rounding next to them
    # still works, and leaves them for the user to sort out.
    points = np.insert(SQUARE, 2, SQUARE[2], axis=0)
    rounded = round_corner(points, np.full(5, 12.0), 1, 20.0)
    duplicates = np.all(rounded.points == SQUARE[2], axis=1)
    assert np.count_nonzero(duplicates) == 2
