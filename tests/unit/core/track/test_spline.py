"""Tests for mlracecar.core.track.spline: examples, a reference implementation, and properties."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlracecar.core.geometry import FloatArray, cross, norm
from mlracecar.core.track.spline import ALPHA, DEFAULT_SPACING, ClosedSpline
from strategies import track_like_points


def circle(count: int, radius: float = 50.0, clockwise: bool = False) -> FloatArray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    if clockwise:
        angles = -angles
    return radius * np.column_stack([np.cos(angles), np.sin(angles)])


def reference_position(points: FloatArray, piece: int, fraction: float) -> FloatArray:
    """Textbook periodic cubic spline, written independently of the implementation under test.

    Loops instead of vectorizing, and uses the classic form
    ``y = A y_i + B y_{i+1} + ((A^3 - A) m_i + (B^3 - B) m_{i+1}) h^2 / 6``.
    """
    count = len(points)
    h = [math.dist(points[i], points[(i + 1) % count]) ** ALPHA for i in range(count)]
    system = np.zeros((count, count))
    rhs = np.zeros((count, 2))
    for i in range(count):
        before, after = h[i - 1], h[i]
        system[i, i - 1] += before
        system[i, i] += 2 * (before + after)
        system[i, (i + 1) % count] += after
        rhs[i] = 6 * (
            (points[(i + 1) % count] - points[i]) / after - (points[i] - points[i - 1]) / before
        )
    second = np.linalg.solve(system, rhs)
    weight_after = fraction
    weight_before = 1 - fraction
    result: FloatArray = (
        weight_before * points[piece]
        + weight_after * points[(piece + 1) % count]
        + (
            (weight_before**3 - weight_before) * second[piece]
            + (weight_after**3 - weight_after) * second[(piece + 1) % count]
        )
        * h[piece] ** 2
        / 6
    )
    return result


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([[0, 0], [1, 0]], "at least 3 control points"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "at least 3 control points"),
        ([[0, 0], [1, 0], [1, 0], [0, 1]], r"coincide .* indices \[1\]"),
        ([[0, 0], [1, 0], [0, 1], [0, 0]], r"coincide .* indices \[3\]"),  # last = first
        ([[0, 0], [1, 0], [1, 1e-248], [0, 1]], r"coincide .* indices \[1\]"),  # too close
    ],
)
def test_rejects_invalid_control_points(points: list[list[float]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ClosedSpline.through(points)


def test_has_one_piece_per_control_point() -> None:
    spline = ClosedSpline.through(circle(7))
    assert spline.coefficients.shape == (7, 4, 2)


# --------------------------------------------------------------------------- #
# Shape
# --------------------------------------------------------------------------- #


@given(track_like_points())
def test_passes_through_every_control_point(points: FloatArray) -> None:
    spline = ClosedSpline.through(points)
    pieces = np.arange(len(points))
    np.testing.assert_allclose(spline.position(pieces, np.zeros(len(points))), points, atol=1e-9)
    # The end of each piece meets the next control point, closing the loop at point 0.
    np.testing.assert_allclose(
        spline.position(pieces, np.ones(len(points))), np.roll(points, -1, axis=0), atol=1e-9
    )


@given(track_like_points(), st.data())
def test_matches_textbook_formulation(points: FloatArray, data: st.DataObject) -> None:
    spline = ClosedSpline.through(points)
    piece = data.draw(st.integers(min_value=0, max_value=len(points) - 1))
    fraction = data.draw(st.floats(min_value=0, max_value=1))
    np.testing.assert_allclose(
        spline.position(piece, fraction), reference_position(points, piece, fraction), atol=1e-8
    )


@given(track_like_points())
def test_direction_and_curvature_are_continuous_between_pieces(points: FloatArray) -> None:
    spline = ClosedSpline.through(points)
    ends = np.arange(len(points))
    starts = (ends + 1) % len(points)
    ones, zeros = np.ones(len(points)), np.zeros(len(points))

    def direction(piece: np.ndarray, fraction: np.ndarray) -> FloatArray:
        velocity = spline.velocity(piece, fraction)
        unit: FloatArray = velocity / norm(velocity)[:, None]
        return unit

    np.testing.assert_allclose(direction(ends, ones), direction(starts, zeros), atol=1e-9)
    # C2: the bend at the end of one piece equals the bend at the start of the next.
    end_curvature = spline.curvature(ends, ones)
    start_curvature = spline.curvature(starts, zeros)
    np.testing.assert_allclose(end_curvature, start_curvature, rtol=1e-6, atol=1e-9)


@given(track_like_points())
def test_is_at_least_as_long_as_the_control_polygon(points: FloatArray) -> None:
    perimeter = float(norm(np.roll(points, -1, axis=0) - points).sum())
    assert ClosedSpline.through(points).length >= perimeter


# --------------------------------------------------------------------------- #
# Curvature
# --------------------------------------------------------------------------- #


def test_curvature_of_a_circle_is_within_one_percent() -> None:
    # 20 control points per full circle: the minimum for 1% (16 points give 1.3%, 8 give 5.7%).
    radius = 50.0
    centerline = ClosedSpline.through(circle(20, radius)).resample()
    np.testing.assert_allclose(centerline.curvature, 1 / radius, rtol=0.01)


def test_curvature_sign_follows_turn_direction() -> None:
    assert np.all(ClosedSpline.through(circle(12)).resample().curvature > 0)  # left turns
    assert np.all(ClosedSpline.through(circle(12, clockwise=True)).resample().curvature < 0)


@given(track_like_points())
def test_curvature_matches_how_fast_the_direction_turns(points: FloatArray) -> None:
    centerline = ClosedSpline.through(points).resample(0.25)
    heading = np.arctan2(centerline.tangent[:, 1], centerline.tangent[:, 0])
    turn = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))  # wrapped heading change
    step = np.roll(centerline.arc_length, -1) - centerline.arc_length
    step[-1] += centerline.length
    average_curvature = (centerline.curvature + np.roll(centerline.curvature, -1)) / 2
    # A finite difference over 0.25 m is itself an approximation, worst where the bend changes
    # fastest (it shrinks with the step size). Allow 1% of the track's sharpest bend.
    sharpest = float(np.abs(centerline.curvature).max())
    np.testing.assert_allclose(turn / step, average_curvature, rtol=0, atol=0.01 * sharpest)


@given(track_like_points())
def test_a_simple_counter_clockwise_loop_turns_once(points: FloatArray) -> None:
    centerline = ClosedSpline.through(points).resample()
    total_turn = float(np.sum(centerline.curvature) * centerline.length / len(centerline.points))
    assert math.isclose(total_turn, 2 * math.pi, rel_tol=0.01)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


@given(track_like_points(), st.floats(min_value=0.1, max_value=1.0))
def test_resampled_spacing_is_within_two_percent(points: FloatArray, spacing: float) -> None:
    # Gaps are measured as straight lines, which cut corners: fine at realistic spacings, but
    # at several metres per sample a tight hairpin makes the straight line ~2% shorter.
    centerline = ClosedSpline.through(points).resample(spacing)
    gaps = norm(np.roll(centerline.points, -1, axis=0) - centerline.points)
    np.testing.assert_allclose(gaps, spacing, rtol=0.02)


@given(track_like_points())
def test_resampled_fields_are_consistent(points: FloatArray) -> None:
    spline = ClosedSpline.through(points)
    centerline = spline.resample()
    n = len(centerline.points)

    assert centerline.length == spline.length
    assert centerline.arc_length[0] == 0
    assert np.all(np.diff(centerline.arc_length) > 0)
    assert centerline.arc_length[-1] < centerline.length
    np.testing.assert_allclose(centerline.points[0], points[0], atol=1e-9)
    np.testing.assert_allclose(norm(centerline.tangent), np.ones(n))
    np.testing.assert_allclose(cross(centerline.tangent, centerline.normal), np.ones(n))  # left
    np.testing.assert_allclose(
        spline.position(centerline.piece, centerline.fraction), centerline.points
    )
    assert np.all((centerline.fraction >= 0) & (centerline.fraction < 1))
    assert np.all(np.diff(centerline.piece) >= 0)


@given(track_like_points(), st.lists(st.floats(-500, 1500), min_size=1, max_size=10))
def test_locate_and_arc_length_at_are_inverses(points: FloatArray, distances: list[float]) -> None:
    spline = ClosedSpline.through(points)
    piece, fraction = spline.locate(distances)
    assert np.all((piece >= 0) & (piece < len(points)))
    np.testing.assert_allclose(
        spline.arc_length_at(piece, fraction), np.mod(distances, spline.length), atol=1e-6
    )


def test_arc_length_at_control_points_starts_at_zero_and_increases() -> None:
    spline = ClosedSpline.through(circle(8))
    at_points = spline.arc_length_at(np.arange(8), 0.0)
    assert at_points[0] == 0
    np.testing.assert_allclose(np.diff(at_points), spline.length / 8, rtol=1e-6)


def test_default_spacing_is_half_a_metre() -> None:
    centerline = ClosedSpline.through(circle(20)).resample()
    assert DEFAULT_SPACING == 0.5
    assert math.isclose(centerline.length / len(centerline.points), 0.5, rel_tol=0.01)


def test_very_coarse_spacing_still_gives_three_samples() -> None:
    assert len(ClosedSpline.through(circle(4, radius=1.0)).resample(100.0).points) == 3


@pytest.mark.parametrize("spacing", [0.0, -1.0, float("nan")])
def test_rejects_non_positive_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="spacing must be positive"):
        ClosedSpline.through(circle(8)).resample(spacing)
