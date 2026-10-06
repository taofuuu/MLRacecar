"""Tests for mlracecar.core.track.model: widths, edges, checkpoints, and the starting grid."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlracecar.core.geometry import (
    FloatArray,
    cross,
    norm,
    project_onto_polyline,
    unit_vector,
    wrap_angle,
)
from mlracecar.core.track.model import MIN_CHECKPOINTS, GridLayout, Track
from strategies import track_like_points, track_widths

OVAL = 60 * np.column_stack(
    [2 * np.cos(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
     np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))]
)  # fmt: skip  # 240 m x 120 m, counter-clockwise


@st.composite
def track_inputs(draw: st.DrawFn) -> tuple[FloatArray, FloatArray]:
    """Control points (either driving direction) and widths for a track-like shape.

    Tests draw these small inputs rather than whole `Track` objects, so a failing example
    prints a few numbers instead of every derived array.
    """
    points = draw(track_like_points(clockwise=None))
    return points, draw(track_widths(len(points)))


def polygon_area(vertices: FloatArray) -> float:
    """Signed area (shoelace formula): positive when the vertices run counter-clockwise."""
    return float(np.sum(cross(vertices, np.roll(vertices, -1, axis=0))) / 2)


def rectangles_overlap(
    center_a: FloatArray, heading_a: float, center_b: FloatArray, heading_b: float, size: GridLayout
) -> bool:
    """Separating-axis test for two car-sized rectangles."""
    half = np.array([size.car_length, size.car_width]) / 2
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * half

    def outline(center: FloatArray, heading: float) -> FloatArray:
        c, s = math.cos(heading), math.sin(heading)
        result: FloatArray = center + corners @ np.array([[c, s], [-s, c]])
        return result

    a, b = outline(center_a, heading_a), outline(center_b, heading_b)
    for heading in (heading_a, heading_b):
        for axis in (unit_vector(heading), unit_vector(heading + math.pi / 2)):
            project_a, project_b = a @ axis, b @ axis
            if project_a.max() < project_b.min() or project_b.max() < project_a.min():
                return False  # found a gap: they don't overlap
    return True


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("widths", "kwargs", "message"),
    [
        ([10.0] * 11, {}, r"one width per control point \(12\)"),
        ([10.0] * 11 + [0.0], {}, "widths must be positive"),
        ([10.0] * 11 + [-3.0], {}, "widths must be positive"),
        ([10.0] * 12, {"checkpoint_spacing": 0.0}, "checkpoint_spacing must be positive"),
    ],
)
def test_build_rejects_invalid_input(
    widths: list[float], kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Track.build(OVAL, widths, **kwargs)


def test_repr_is_a_short_summary() -> None:
    track = Track.build(OVAL, [12.0] * 12)
    assert repr(track).startswith(f"Track(length={track.length:.1f} m, 12 control points, ")


# --------------------------------------------------------------------------- #
# Widths and edges
# --------------------------------------------------------------------------- #


@given(track_inputs())
def test_edges_are_offset_half_the_width_along_the_normal(
    inputs: tuple[FloatArray, FloatArray],
) -> None:
    track = Track.build(*inputs)
    center, normal = track.centerline.points, track.centerline.normal
    half = track.width / 2
    np.testing.assert_allclose(norm(track.left - center), half)
    np.testing.assert_allclose(norm(track.right - center), half)
    np.testing.assert_allclose(track.left - center, half[:, None] * normal, atol=1e-9)
    np.testing.assert_allclose(track.right - center, -half[:, None] * normal, atol=1e-9)


@given(track_like_points(clockwise=None), st.data())
def test_left_edge_is_inside_a_counter_clockwise_track_and_outside_a_clockwise_one(
    points: FloatArray, data: st.DataObject
) -> None:
    track = Track.build(points, data.draw(track_widths(len(points))))
    counter_clockwise = polygon_area(points) > 0
    left_area, right_area = abs(polygon_area(track.left)), abs(polygon_area(track.right))
    assert (left_area < right_area) is counter_clockwise


@given(track_like_points(), st.data())
def test_width_equals_the_control_width_at_each_control_point(
    points: FloatArray, data: st.DataObject
) -> None:
    widths = data.draw(track_widths(len(points)))
    track = Track.build(points, widths)
    at_points = track.spline.arc_length_at(np.arange(len(points)), 0.0)
    np.testing.assert_allclose(track.width_at(at_points), widths, rtol=1e-6)


@given(track_inputs())
def test_width_never_leaves_the_range_of_its_two_control_widths(
    inputs: tuple[FloatArray, FloatArray],
) -> None:
    track = Track.build(*inputs)
    piece = track.centerline.piece
    start = track.control_widths[piece]
    end = track.control_widths[(piece + 1) % len(track.control_widths)]
    tolerance = 1e-9
    assert np.all(track.width >= np.minimum(start, end) - tolerance)
    assert np.all(track.width <= np.maximum(start, end) + tolerance)


def test_equal_control_widths_give_a_constant_width() -> None:
    np.testing.assert_allclose(Track.build(OVAL, [9.0] * 12).width, 9.0)


# --------------------------------------------------------------------------- #
# Positions along the track
# --------------------------------------------------------------------------- #


def test_pose_at_wraps_around_the_lap_and_shifts_sideways() -> None:
    track = Track.build(OVAL, [12.0] * 12)
    np.testing.assert_allclose(
        track.pose_at(-1.0).position, track.pose_at(track.length - 1).position
    )
    np.testing.assert_allclose(
        track.pose_at(track.length).position, track.pose_at(0.0).position, atol=1e-9
    )
    start = track.pose_at(0.0)
    np.testing.assert_allclose(start.position, OVAL[0], atol=1e-9)
    assert math.isclose(start.heading, math.pi / 2, abs_tol=1e-6)  # heading north at the start
    np.testing.assert_allclose(track.pose_at(0.0, 3.0).position, OVAL[0] + [-3.0, 0.0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


@given(track_inputs(), st.floats(min_value=5.0, max_value=100.0))
def test_checkpoints_are_evenly_spaced_lines_across_the_road(
    inputs: tuple[FloatArray, FloatArray], spacing: float
) -> None:
    track = Track.build(*inputs, checkpoint_spacing=spacing)
    checkpoints = track.checkpoints
    count = len(checkpoints.arc_length)
    assert count == max(MIN_CHECKPOINTS, round(track.length / spacing))
    assert checkpoints.arc_length[0] == 0  # checkpoint 0 is the start/finish line
    np.testing.assert_allclose(np.diff(checkpoints.arc_length), track.length / count)

    center = track.pose_at(checkpoints.arc_length)
    half_width = track.width_at(checkpoints.arc_length) / 2
    np.testing.assert_allclose(norm(checkpoints.left - center.position), half_width, rtol=1e-9)
    np.testing.assert_allclose(norm(checkpoints.right - center.position), half_width, rtol=1e-9)
    across = checkpoints.left - checkpoints.right
    along = unit_vector(center.heading)
    np.testing.assert_allclose(np.einsum("ni,ni->n", across, along), 0, atol=1e-6)  # perpendicular


def test_start_finish_line_is_at_the_first_control_point() -> None:
    line = Track.build(OVAL, [12.0] * 12).checkpoints
    np.testing.assert_allclose((line.left[0] + line.right[0]) / 2, OVAL[0], atol=1e-9)


def test_a_short_track_still_gets_the_minimum_number_of_checkpoints() -> None:
    small = Track.build(OVAL / 10, [3.0] * 12, checkpoint_spacing=500.0)
    assert len(small.checkpoints.arc_length) == MIN_CHECKPOINTS


# --------------------------------------------------------------------------- #
# Starting grid
# --------------------------------------------------------------------------- #


@given(track_inputs(), st.integers(min_value=0, max_value=20))
def test_grid_cars_never_overlap(inputs: tuple[FloatArray, FloatArray], count: int) -> None:
    track = Track.build(*inputs)
    grid = track.start_grid(count)
    layout = GridLayout()
    for i in range(count):
        for j in range(i + 1, count):
            assert not rectangles_overlap(
                grid.position[i], grid.heading[i], grid.position[j], grid.heading[j], layout
            ), f"cars {i} and {j} overlap"


@given(track_inputs(), st.integers(min_value=1, max_value=20))
def test_grid_cars_face_the_driving_direction_behind_the_start_line(
    inputs: tuple[FloatArray, FloatArray], count: int
) -> None:
    track = Track.build(*inputs)
    layout = GridLayout()
    grid = track.start_grid(count, layout)
    on_center = project_onto_polyline(grid.position, track.centerline.points, closed=True)

    # Behind the line: the pole car's center is exactly first_gap + half a car length back.
    distance_behind = track.length - on_center.arc_length
    pole_back = layout.first_gap + layout.car_length / 2
    np.testing.assert_allclose(distance_behind[0], pole_back, atol=0.05)
    np.testing.assert_allclose(np.diff(distance_behind), layout.car_length + layout.gap, atol=0.05)

    # Facing the driving direction: heading matches the road's direction where the car stands
    # (found by projecting onto the sampled centerline, independently of how the grid was built).
    road_heading = track.pose_at(on_center.arc_length).heading
    np.testing.assert_allclose(wrap_angle(grid.heading - road_heading), 0, atol=1e-3)

    # Staggered: pole position on the left, then alternating, with every car on the road.
    expected_side = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    np.testing.assert_array_equal(np.sign(on_center.offset), expected_side)
    room = track.width_at(on_center.arc_length) / 2 - layout.car_width / 2
    assert np.all(np.abs(on_center.offset) <= room + 0.01)


def test_narrow_road_moves_lanes_inwards_to_keep_cars_on_it() -> None:
    narrow = Track.build(OVAL, [5.0] * 12)  # quarter width 1.25 m, but a 2 m car needs 1.5 m
    grid = narrow.start_grid(2)
    offsets = project_onto_polyline(grid.position, narrow.centerline.points, closed=True).offset
    np.testing.assert_allclose(np.abs(offsets), 1.25, atol=0.01)  # min(5/4, (5-2)/2) = 1.25
    very_narrow = Track.build(OVAL, [1.5] * 12)  # narrower than a car: lanes collapse to center
    grid = very_narrow.start_grid(2)
    centered = project_onto_polyline(grid.position, very_narrow.centerline.points, closed=True)
    np.testing.assert_allclose(centered.offset, 0, atol=0.01)


def test_empty_grid_and_negative_count() -> None:
    track = Track.build(OVAL, [12.0] * 12)
    assert track.start_grid(0).position.shape == (0, 2)
    with pytest.raises(ValueError, match="count must not be negative"):
        track.start_grid(-1)
