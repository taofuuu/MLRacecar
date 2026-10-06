"""Tests for mlracecar.core.geometry: worked examples plus properties checked on random inputs."""

import math

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mlracecar.core.geometry import (
    cast_rays,
    cross,
    norm,
    polyline_crossings,
    project_onto_polyline,
    ray_segment_distances,
    rotate,
    segments_intersect,
    self_intersections,
    unit_vector,
    wrap_angle,
)

# --------------------------------------------------------------------------- #
# Strategies: coordinates on a race-track scale (metres), no NaN or infinity.
# --------------------------------------------------------------------------- #

coordinate = st.floats(min_value=-1_000, max_value=1_000, allow_nan=False, allow_infinity=False)
angle = st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
point = st.tuples(coordinate, coordinate).map(lambda xy: np.array(xy, dtype=np.float64))


def points(count: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(np.float64, (count, 2), elements=coordinate)


polyline = st.integers(min_value=2, max_value=12).flatmap(points)


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #


def test_cross_sign_follows_turn_direction() -> None:
    east, north = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert cross(east, north) == 1.0  # north is counter-clockwise (left) of east
    assert cross(north, east) == -1.0
    assert cross(east, 2 * east) == 0.0


def test_norm_is_vector_length() -> None:
    np.testing.assert_allclose(norm([[3.0, 4.0], [0.0, 0.0]]), [5.0, 0.0])


@pytest.mark.parametrize("bad", [5.0, [1.0, 2.0, 3.0], [[1.0], [2.0]]])
def test_vector_functions_reject_wrong_shapes(bad: object) -> None:
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 2\)"):
        norm(bad)  # type: ignore[arg-type]


@given(angle)
def test_wrap_angle_stays_in_range_and_points_the_same_way(a: float) -> None:
    wrapped = float(wrap_angle(a))
    assert -math.pi <= wrapped < math.pi
    assert math.isclose(math.cos(wrapped), math.cos(a), abs_tol=1e-9)
    assert math.isclose(math.sin(wrapped), math.sin(a), abs_tol=1e-9)


def test_wrap_angle_folds_rounding_at_plus_pi() -> None:
    # Just below -pi, the modulo rounds to exactly 2*pi, which would map to +pi.
    assert float(wrap_angle(-math.pi - 1e-17)) == -math.pi
    np.testing.assert_allclose(
        wrap_angle([0.0, 3 * math.pi, -3 * math.pi]), [0.0, -math.pi, -math.pi]
    )


def test_unit_vector_points_along_angle() -> None:
    np.testing.assert_allclose(
        unit_vector([0.0, math.pi / 2]), [[1.0, 0.0], [0.0, 1.0]], atol=1e-15
    )


@given(point, angle)
def test_rotate_keeps_length_and_is_undone_by_rotating_back(v: np.ndarray, a: float) -> None:
    rotated = rotate(v, a)
    assert math.isclose(float(norm(rotated)), float(norm(v)), rel_tol=1e-12, abs_tol=1e-9)
    np.testing.assert_allclose(rotate(rotated, -a), v, atol=1e-9)


def test_rotate_quarter_turn_counter_clockwise() -> None:
    np.testing.assert_allclose(rotate([1.0, 0.0], math.pi / 2), [0.0, 1.0], atol=1e-15)


def test_rotate_broadcasts_one_angle_per_vector() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        rotate(vectors, [0.0, math.pi]), [[1.0, 0.0], [-1.0, 0.0]], atol=1e-15
    )


# --------------------------------------------------------------------------- #
# Segment intersection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("a_start", "a_end", "b_start", "b_end", "expected"),
    [
        pytest.param((0, 0), (2, 2), (0, 2), (2, 0), True, id="cross-in-the-middle"),
        pytest.param((0, 0), (1, 0), (1, 0), (1, 1), True, id="touch-at-endpoint"),
        pytest.param((0, 0), (1, 0), (2, -1), (2, 1), False, id="would-cross-if-longer"),
        pytest.param((0, 0), (1, 0), (0, 1), (1, 1), False, id="parallel"),
        pytest.param((0, 0), (2, 0), (1, 0), (3, 0), False, id="collinear-overlap-is-parallel"),
        pytest.param((0, 0), (0, 0), (-1, 0), (1, 0), False, id="zero-length"),
    ],
)
def test_segments_intersect_examples(
    a_start: tuple[float, float],
    a_end: tuple[float, float],
    b_start: tuple[float, float],
    b_end: tuple[float, float],
    expected: bool,
) -> None:
    assert bool(segments_intersect(a_start, a_end, b_start, b_end)) is expected


@given(point, point, point, point)
def test_segments_intersect_is_symmetric(
    a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray
) -> None:
    assert bool(segments_intersect(a1, a2, b1, b2)) == bool(segments_intersect(b1, b2, a1, a2))


def test_segments_intersect_broadcasts_pairwise() -> None:
    starts = np.array([[0.0, 0.0], [0.0, 5.0]])
    ends = np.array([[2.0, 2.0], [1.0, 5.0]])
    result = segments_intersect(starts[:, None], ends[:, None], starts[None, :], ends[None, :])
    assert result.shape == (2, 2)
    assert not result[0, 1]
    assert not result[1, 0]


# --------------------------------------------------------------------------- #
# Raycasts
# --------------------------------------------------------------------------- #


def test_ray_hits_wall_ahead_and_misses_wall_behind() -> None:
    wall_start, wall_end = (
        np.array([[5.0, -1.0], [-5.0, -1.0]]),
        np.array([[5.0, 1.0], [-5.0, 1.0]]),
    )
    distances = ray_segment_distances([0.0, 0.0], [2.0, 0.0], wall_start, wall_end)
    # Distance is in metres even though the direction vector has length 2.
    np.testing.assert_allclose(distances, [5.0, np.inf])


def test_ray_parallel_to_segment_misses() -> None:
    assert ray_segment_distances([0, 0], [1, 0], [1, 0], [3, 0]) == np.inf


@given(point, point, point, st.floats(min_value=0.0, max_value=1.0))
def test_ray_aimed_at_a_segment_point_hits_it_at_that_distance(
    origin: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray, fraction: float
) -> None:
    target = seg_start + fraction * (seg_end - seg_start)
    direction = target - origin
    segment = seg_end - seg_start
    # Skip near-degenerate cases: ray almost parallel to the segment, or tiny vectors.
    assume(float(norm(direction)) > 1e-3 and float(norm(segment)) > 1e-3)
    assume(abs(float(cross(direction, segment))) > 1e-3 * float(norm(direction) * norm(segment)))

    distance = float(ray_segment_distances(origin, direction, seg_start, seg_end))
    assert math.isclose(distance, float(norm(direction)), rel_tol=1e-6, abs_tol=1e-6)


@given(point, point, point, point)
def test_ray_through_a_shared_corner_never_slips_between_segments(
    origin: np.ndarray, before: np.ndarray, corner: np.ndarray, after: np.ndarray
) -> None:
    # A boundary made of two connected segments, and a ray aimed exactly at their shared corner.
    direction = corner - origin
    assume(float(norm(direction)) > 1e-3)
    for other_end in (before, after):
        segment = corner - other_end
        assume(float(norm(segment)) > 1e-3)
        assume(
            abs(float(cross(direction, segment))) > 1e-3 * float(norm(direction) * norm(segment))
        )

    distance = cast_rays([origin], [direction], [before, corner], [corner, after])[0]
    assert math.isclose(distance, float(norm(direction)), rel_tol=1e-6, abs_tol=1e-6)


def test_cast_rays_reports_nearest_wall_and_caps_at_range() -> None:
    walls_start = np.array([[3.0, -1.0], [6.0, -1.0], [-1.0, 20.0]])
    walls_end = np.array([[3.0, 1.0], [6.0, 1.0], [1.0, 20.0]])
    origins = np.zeros((3, 2))
    directions = unit_vector([0.0, math.pi / 2, math.pi])  # east, north, west
    distances = cast_rays(origins, directions, walls_start, walls_end, max_distance=10.0)
    np.testing.assert_allclose(distances, [3.0, 10.0, 10.0])


def test_cast_rays_without_segments_returns_range() -> None:
    empty = np.empty((0, 2))
    np.testing.assert_array_equal(
        cast_rays(np.zeros((2, 2)), np.ones((2, 2)), empty, empty, 7.0), [7.0, 7.0]
    )


# --------------------------------------------------------------------------- #
# Projection onto a polyline
# --------------------------------------------------------------------------- #

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])  # counter-clockwise


def test_projection_on_closed_square() -> None:
    result = project_onto_polyline([[5.0, 2.0], [12.0, 5.0], [5.0, -3.0]], SQUARE, closed=True)
    np.testing.assert_allclose(result.point, [[5.0, 0.0], [10.0, 5.0], [5.0, 0.0]])
    np.testing.assert_array_equal(result.segment, [0, 1, 0])
    np.testing.assert_allclose(result.arc_length, [5.0, 15.0, 5.0])
    # Inside a counter-clockwise loop is to the left of the direction of travel.
    np.testing.assert_allclose(result.offset, [2.0, -2.0, -3.0])


def test_projection_uses_closing_segment_only_when_closed() -> None:
    query = [[-1.0, 6.0]]  # beside the edge from (0, 10) back to (0, 0), nearer (0, 10)
    closed = project_onto_polyline(query, SQUARE, closed=True)
    opened = project_onto_polyline(query, SQUARE, closed=False)
    assert closed.segment[0] == 3
    assert math.isclose(closed.arc_length[0], 34.0)
    assert closed.offset[0] == -1.0  # outside a counter-clockwise loop is on the right
    assert opened.segment[0] == 2  # without the closing edge, the corner (0, 10) is nearest
    np.testing.assert_allclose(opened.point, [[0.0, 10.0]])


def test_projection_beyond_the_end_keeps_the_distance() -> None:
    # In line with the only segment: no side is defined, but the distance must not be lost.
    result = project_onto_polyline([[3.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], closed=False)
    np.testing.assert_allclose(result.point, [[1.0, 0.0]])
    assert result.offset[0] == 2.0


def test_projection_handles_repeated_vertices() -> None:
    line = [[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]]  # first segment has zero length
    result = project_onto_polyline([[2.0, 1.0]], line, closed=False)
    np.testing.assert_allclose(result.point, [[2.0, 0.0]])
    assert math.isclose(result.arc_length[0], 2.0)


@pytest.mark.parametrize(
    ("query", "vertices", "message"),
    [
        ([1.0, 2.0], SQUARE, r"points must have shape \(N, 2\)"),
        ([[1.0, 2.0]], [[0.0, 0.0]], r"V >= 2"),
    ],
)
def test_projection_rejects_bad_shapes(query: object, vertices: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        project_onto_polyline(query, vertices, closed=False)  # type: ignore[arg-type]


@given(points(5), polyline, st.booleans())
def test_projection_is_never_farther_than_any_vertex(
    query: np.ndarray, vertices: np.ndarray, closed: bool
) -> None:
    result = project_onto_polyline(query, vertices, closed=closed)
    to_vertices = np.linalg.norm(query[:, None, :] - vertices[None, :, :], axis=-1).min(axis=1)
    assert np.all(np.abs(result.offset) <= to_vertices + 1e-9)
    np.testing.assert_allclose(np.abs(result.offset), norm(query - result.point), atol=1e-9)


@given(points(5), polyline, st.booleans())
def test_projection_arc_length_stays_within_polyline(
    query: np.ndarray, vertices: np.ndarray, closed: bool
) -> None:
    result = project_onto_polyline(query, vertices, closed=closed)
    segment_ends = np.roll(vertices, -1, axis=0) if closed else vertices[1:]
    segment_starts = vertices if closed else vertices[:-1]
    total = norm(segment_ends - segment_starts).sum()
    assert np.all(result.arc_length >= 0)
    assert np.all(result.arc_length <= total + 1e-9)


@given(points(6), polyline, st.booleans())
def test_projection_of_a_batch_matches_projecting_one_by_one(
    query: np.ndarray, vertices: np.ndarray, closed: bool
) -> None:
    batch = project_onto_polyline(query, vertices, closed=closed)
    for n in range(len(query)):
        single = project_onto_polyline(query[n : n + 1], vertices, closed=closed)
        for batch_field, single_field in zip(batch, single, strict=True):
            np.testing.assert_array_equal(batch_field[n : n + 1], single_field)


# --------------------------------------------------------------------------- #
# Self-intersections
# --------------------------------------------------------------------------- #


def test_figure_eight_crosses_itself() -> None:
    bow_tie = [[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]
    np.testing.assert_array_equal(self_intersections(bow_tie, closed=True), [[0, 2]])


def test_open_zigzag_that_doubles_back_is_detected() -> None:
    zigzag = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [5.0, -5.0]]
    np.testing.assert_array_equal(self_intersections(zigzag, closed=False), [[0, 2]])


@given(
    st.integers(min_value=3, max_value=600),
    st.floats(min_value=1.0, max_value=500.0),
    point,
    angle,
)
def test_convex_polygons_never_cross_themselves(
    sides: int, radius: float, center: np.ndarray, rotation: float
) -> None:
    corners = center + radius * unit_vector(
        rotation + np.linspace(0, 2 * np.pi, sides, endpoint=False)
    )
    # Up to 600 sides also exercises several comparison blocks (64 segments each).
    assert self_intersections(corners, closed=True).shape == (0, 2)


def brute_force_pairs(a: np.ndarray, b: np.ndarray, closed: bool) -> set[tuple[int, int]]:
    """Every segment pair that touches, found the slow way (no broad phase)."""
    a_end = np.roll(a, -1, axis=0) if closed else a[1:]
    b_end = np.roll(b, -1, axis=0) if closed else b[1:]
    a_start, b_start = (a, b) if closed else (a[:-1], b[:-1])
    pairs = set()
    for i in range(len(a_start)):
        for j in range(len(b_start)):
            if segments_intersect(a_start[i], a_end[i], b_start[j], b_end[j]):
                pairs.add((i, j))
    return pairs


@given(polyline, polyline, st.booleans())
def test_polyline_crossings_matches_brute_force(
    first: np.ndarray, second: np.ndarray, closed: bool
) -> None:
    found = {tuple(pair) for pair in polyline_crossings(first, second, closed=closed).tolist()}
    assert found == brute_force_pairs(first, second, closed)


@given(polyline, st.booleans())
def test_self_intersections_matches_brute_force(vertices: np.ndarray, closed: bool) -> None:
    count = len(vertices) if closed else len(vertices) - 1
    expected = {
        (i, j)
        for i, j in brute_force_pairs(vertices, vertices, closed)
        if j > i + 1 and not (closed and i == 0 and j == count - 1)
    }
    found = {tuple(pair) for pair in self_intersections(vertices, closed=closed).tolist()}
    assert found == expected


def test_polyline_crossings_between_overlapping_and_distant_squares() -> None:
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    shifted = square + np.array([5.0, 5.0])  # overlaps the top-right quarter
    np.testing.assert_array_equal(
        polyline_crossings(square, shifted, closed=True), [[1, 0], [2, 3]]
    )
    far_away = square + np.array([100.0, 100.0])  # the broad phase skips every block
    assert polyline_crossings(square, far_away, closed=True).shape == (0, 2)
