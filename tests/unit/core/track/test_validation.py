"""Tests for mlracecar.core.track.validation: one failing example per rule, plus robustness."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from circuits import gp_circuit
from mlracecar.core.geometry import FloatArray, project_onto_polyline, self_intersections
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import (
    ControlPointAt,
    IssueCode,
    Severity,
    Spot,
    Stretch,
    ValidationIssue,
    ValidationRules,
    has_errors,
    validate,
)


def ellipse(count: int, a: float, b: float) -> FloatArray:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.column_stack([a * np.cos(angles), b * np.sin(angles)])


def same_width(points: FloatArray, width: float) -> FloatArray:
    return np.full(len(points), width)


def codes(issues: list[ValidationIssue]) -> list[IssueCode]:
    return [issue.code for issue in issues]


# Sample tracks: valid as drawn.
OVAL = ellipse(12, 120, 60)
PLAYGROUND = np.array(
    [[0, 0], [60, -10], [110, 0], [135, 35], [105, 62], [72, 42], [45, 72], [5, 62], [-22, 30]],
    dtype=np.float64,
)

# Five unevenly spaced points whose control polygon is simple, but whose C2 spline overshoots
# into a small loop (found by searching random shapes; see ADR-0010).
OVERSHOOT = np.array([[5, 33], [-33, -23], [4, -42], [103, -105], [22, -19]], dtype=np.float64)


def paperclip(radius: float, straight: float = 150.0) -> FloatArray:
    """Two straights joined by 180-degree hairpins, with dots like a person would place them."""
    along = np.arange(-straight / 2, straight / 2, radius / 2)
    halves = np.linspace(-np.pi / 2, np.pi / 2, 9)[:-1]
    bottom = np.column_stack([along, np.full(len(along), -radius)])
    right = np.column_stack([straight / 2 + radius * np.cos(halves), radius * np.sin(halves)])
    top = np.column_stack([-along, np.full(len(along), radius)])
    left = np.column_stack([-straight / 2 - radius * np.cos(halves), -radius * np.sin(halves)])
    return np.vstack([bottom, right, top, left])


def peanut(waist_gap: float) -> FloatArray:
    """A 220 m x 140 m oval pinched in the middle until the waist is ``waist_gap`` metres wide."""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    x = 110 * np.cos(angles)
    pinch = 1 - (1 - waist_gap / 140) * np.exp(-((x / 45) ** 2))
    return np.column_stack([x, 70 * np.sin(angles) * pinch])


# --------------------------------------------------------------------------- #
# Valid tracks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("points", "width"),
    [(OVAL, 12.0), (OVAL[::-1].copy(), 12.0), (PLAYGROUND, 9.0)],
    ids=["oval", "clockwise", "playground"],
)
def test_sample_tracks_have_no_issues(points: FloatArray, width: float) -> None:
    assert validate(points, same_width(points, width)) == []


# --------------------------------------------------------------------------- #
# Input problems
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("count", [1, 2, 3])
def test_too_few_points(count: int) -> None:
    points = ellipse(count, 80, 80)
    issues = validate(points, same_width(points, 10.0))
    assert codes(issues) == [IssueCode.TOO_FEW_POINTS]
    assert issues[0].location is None
    assert "at least 4 points" in issues[0].message


def test_point_not_finite() -> None:
    points = OVAL.copy()
    points[5] = [np.nan, 0.0]
    issues = validate(points, same_width(points, 12.0))
    assert codes(issues) == [IssueCode.POINT_NOT_FINITE]
    assert issues[0].location == ControlPointAt(5)


def test_width_not_positive_reports_every_bad_point() -> None:
    widths = same_width(OVAL, 12.0)
    widths[[2, 7]] = [0.0, np.nan]
    issues = validate(OVAL, widths)
    assert codes(issues) == [IssueCode.WIDTH_NOT_POSITIVE] * 2
    assert [issue.location for issue in issues] == [ControlPointAt(2), ControlPointAt(7)]


@pytest.mark.parametrize(("duplicate", "expected_index"), [(4, 3), (0, 11)])
def test_points_coincide(duplicate: int, expected_index: int) -> None:
    points = OVAL.copy()
    points[expected_index] = points[duplicate]  # point i on top of the next one (wrapping)
    issues = validate(points, same_width(points, 12.0))
    assert codes(issues) == [IssueCode.POINTS_COINCIDE]
    assert issues[0].location == ControlPointAt(expected_index)


def test_points_a_hair_apart_coincide_too() -> None:
    points = OVAL.copy()
    points[3] = points[4] + [0.0, 5e-248]  # the curve between them can't be computed
    issues = validate(points, same_width(points, 12.0))
    assert codes(issues) == [IssueCode.POINTS_COINCIDE]


def test_rejects_arrays_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(P, 2\)"):
        validate(OVAL, same_width(OVAL, 12.0)[:-1])


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #


def test_too_narrow_points_at_the_narrow_control_point() -> None:
    widths = same_width(OVAL, 12.0)
    widths[11] = 4.0
    issues = validate(OVAL, widths)
    assert codes(issues) == [IssueCode.TOO_NARROW]
    assert issues[0].location == ControlPointAt(11)


def test_neighbouring_narrow_points_are_reported_together() -> None:
    widths = same_width(OVAL, 12.0)
    widths[3:6] = [5.0, 4.0, 5.5]
    issues = validate(OVAL, widths)
    assert codes(issues) == [IssueCode.TOO_NARROW]
    assert "points 3 to 5 is as narrow as 4.0 m" in issues[0].message
    stretch = issues[0].location
    assert isinstance(stretch, Stretch)
    spline = Track.build(OVAL, widths).spline
    np.testing.assert_allclose(stretch, spline.arc_length_at([3, 5], 0.0))


def test_narrow_points_on_both_sides_of_the_start_are_one_group() -> None:
    widths = same_width(OVAL, 12.0)
    widths[[11, 0]] = 4.0
    issues = validate(OVAL, widths)
    assert codes(issues) == [IssueCode.TOO_NARROW]
    assert "points 11 to 0" in issues[0].message


def test_too_short() -> None:
    points = ellipse(10, 14, 12)  # about 82 m around
    issues = validate(points, same_width(points, 6.0))
    assert codes(issues) == [IssueCode.TOO_SHORT]
    assert issues[0].location is None


# --------------------------------------------------------------------------- #
# Bends
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("clockwise", [False, True])
def test_edge_folds_where_the_road_is_too_wide_for_the_bend(clockwise: bool) -> None:
    points = ellipse(12, 120, 30)  # ends of about 7.5 m radius, with a 40 m wide road
    if clockwise:
        points = points[::-1].copy()
    issues = validate(points, same_width(points, 40.0))
    assert codes(issues) == [IssueCode.EDGE_FOLDS] * 2  # one per end of the ellipse

    track = Track.build(points, same_width(points, 40.0))
    tightest = track.centerline.arc_length[np.argmax(np.abs(track.centerline.curvature))]
    stretches = [issue.location for issue in issues]
    assert any(_covers(stretch, tightest, track.length) for stretch in stretches)
    inside = "right" if clockwise else "left"  # the inside of the turn is the folding edge
    assert all(f"the {inside} edge folds" in issue.message for issue in issues)


def test_sharp_inside_corner_is_a_warning() -> None:
    # At 12 m wide, the playground hairpin (radius 6.1 m) leaves its inside edge a 9 cm radius:
    # not a fold, but a near-sharp point.
    issues = validate(PLAYGROUND, same_width(PLAYGROUND, 12.0))
    assert codes(issues) == [IssueCode.SHARP_INSIDE_CORNER]
    assert issues[0].severity is Severity.WARNING
    assert "the right edge come to a sharp point" in issues[0].message  # the hairpin turns right
    assert "inside radius is only 0.1 m" in issues[0].message
    track = Track.build(PLAYGROUND, same_width(PLAYGROUND, 12.0))
    tightest = track.centerline.arc_length[np.argmax(np.abs(track.centerline.curvature))]
    assert _covers(issues[0].location, tightest, track.length)
    # The rule's threshold is adjustable.
    assert (
        validate(PLAYGROUND, same_width(PLAYGROUND, 12.0), ValidationRules(min_inside_radius=0))
        == []
    )


@pytest.mark.parametrize(
    ("width", "code"), [(16.0, IssueCode.SHARP_INSIDE_CORNER), (18.0, IssueCode.EDGE_FOLDS)]
)
def test_each_hairpin_is_reported_once(width: float, code: IssueCode) -> None:
    # A hairpin's bend is tightest at its entry and exit, so the problem stretches come in
    # pieces; pieces that turn the same way and lie close together are one bend.
    points = paperclip(10.0)
    widths = same_width(points, width)
    assert codes(validate(points, widths)) == [code, code]  # two hairpins, one report each
    unmerged = validate(points, widths, ValidationRules(bend_merge_widths=0.0))
    assert len(unmerged) > 2


def test_a_hairpin_split_across_the_start_line_is_still_one_bend() -> None:
    hairpin_apex = len(paperclip(10.0)) - 4  # start the lap in the middle of a hairpin
    points = np.roll(paperclip(10.0), -hairpin_apex, axis=0)
    issues = validate(points, same_width(points, 16.0))
    assert codes(issues) == [IssueCode.SHARP_INSIDE_CORNER] * 2
    wrapping = [
        i.location
        for i in issues
        if isinstance(i.location, Stretch) and i.location.end < i.location.start
    ]
    assert len(wrapping) == 1  # the hairpin at the start runs through the start/finish line


def test_realistic_gp_circuit_only_warns_about_its_slowest_hairpin() -> None:
    circuit = gp_circuit("clean")
    issues = validate(circuit.points, circuit.widths)
    assert codes(issues) == [IssueCode.SHARP_INSIDE_CORNER]
    track = Track.build(circuit.points, circuit.widths)
    hairpin = track.spline.arc_length_at(circuit.corners["T11"], 0.0)
    stretch = issues[0].location
    assert isinstance(stretch, Stretch)
    assert hairpin.min() - 10 <= stretch.start <= stretch.end <= hairpin.max() + 15


def test_gp_circuit_pushed_past_its_limits() -> None:
    circuit = gp_circuit("limit")  # 7.5 m chicane, 24 m wide hairpin, 5 m wide T1
    issues = validate(circuit.points, circuit.widths)
    assert codes(issues) == [IssueCode.TOO_NARROW] + [IssueCode.EDGE_FOLDS] * 3
    first, last = circuit.corners["T1"][0], circuit.corners["T1"][-1]
    assert f"points {first} to {last}" in issues[0].message  # the whole corner, reported once
    chicane_left, chicane_right, hairpin = issues[1:]
    assert "left edge folds" in chicane_left.message  # the chicane's halves turn opposite ways,
    assert "right edge folds" in chicane_right.message  # so they stay two separate bends
    assert "right edge folds" in hairpin.message  # and the hairpin is one bend, not several


def test_tight_but_drivable_bend_is_a_warning() -> None:
    points = ellipse(16, 120, 24.5)  # ends of about 4 m radius; a 6 m road doesn't fold
    issues = validate(points, same_width(points, 6.0))
    assert codes(issues) == [IssueCode.TIGHT_BEND] * 2
    assert all(issue.severity is Severity.WARNING for issue in issues)
    assert not has_errors(issues)


def test_a_bend_that_never_ends_covers_the_whole_lap() -> None:
    points = ellipse(12, 5, 5)  # a 5 m circle: tighter than a car can steer all the way round
    issues = validate(points, same_width(points, 6.0))
    bend = next(issue for issue in issues if issue.code is IssueCode.TIGHT_BEND)
    assert isinstance(bend.location, Stretch)
    assert bend.location.start == 0.0
    assert math.isclose(bend.location.end, 0.0, abs_tol=1e-9)  # all the way back to the start


def test_custom_rules_change_the_thresholds() -> None:
    thin = ellipse(16, 120, 24.5)
    relaxed = ValidationRules(min_width=3.0, min_drivable_radius=3.0)
    assert validate(thin, same_width(thin, 4.0), relaxed) == []
    assert IssueCode.TOO_NARROW in codes(validate(thin, same_width(thin, 4.0)))


# --------------------------------------------------------------------------- #
# Crossings and overlaps
# --------------------------------------------------------------------------- #


def test_overshoot_loop_from_uneven_points_is_reported() -> None:
    assert len(self_intersections(OVERSHOOT, closed=True)) == 0  # the points themselves are fine
    issues = validate(OVERSHOOT, same_width(OVERSHOOT, 6.0))
    crossings = [issue for issue in issues if issue.code is IssueCode.TRACK_CROSSES_ITSELF]
    assert len(crossings) == 1
    spot = crossings[0].location
    assert isinstance(spot, Spot)
    centerline = Track.build(OVERSHOOT, same_width(OVERSHOOT, 6.0)).centerline.points
    distance = project_onto_polyline([spot], centerline, closed=True).offset[0]
    assert abs(distance) < 0.5  # the reported spot is on the track
    # A track that crosses itself also overlaps; that's reported once, as the crossing.
    assert IssueCode.TRACK_OVERLAPS not in codes(issues)


def test_overlapping_parts_of_the_road_are_reported_once() -> None:
    points = peanut(waist_gap=8.0)  # waist centerlines 8 m apart, but the road is 14 m wide
    issues = validate(points, same_width(points, 14.0))
    assert codes(issues) == [IssueCode.TRACK_OVERLAPS]
    spot = issues[0].location
    assert isinstance(spot, Spot)
    assert math.hypot(spot.x, spot.y) < 5.0  # at the waist, in the middle of the track
    assert validate(peanut(waist_gap=30.0), same_width(peanut(30.0), 14.0)) == []  # wide waist


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


@given(
    st.integers(min_value=1, max_value=10).flatmap(
        lambda count: st.tuples(
            arrays(np.float64, (count, 2), elements=st.floats(-300, 300)),
            arrays(np.float64, (count,), elements=st.floats(0, 40)),
        )
    )
)
def test_never_raises_and_always_locates_issues(inputs: tuple[FloatArray, FloatArray]) -> None:
    points, widths = inputs
    issues = validate(points, widths)
    for issue in issues:
        assert issue.message
        match issue.location:
            case ControlPointAt(point_index=index):
                assert 0 <= index < len(points)
            case Stretch(start=start, end=end):
                assert math.isfinite(start)
                assert math.isfinite(end)
            case Spot(x=x, y=y):
                assert math.isfinite(x)
                assert math.isfinite(y)
            case None:
                assert issue.code in {IssueCode.TOO_FEW_POINTS, IssueCode.TOO_SHORT}


def test_has_errors_ignores_warnings() -> None:
    warning = ValidationIssue(Severity.WARNING, IssueCode.TIGHT_BEND, "tight", None)
    error = ValidationIssue(Severity.ERROR, IssueCode.TOO_SHORT, "short", None)
    assert not has_errors([])
    assert not has_errors([warning])
    assert has_errors([warning, error])


def _covers(stretch: object, arc_length: float, length: float) -> bool:
    """Whether a (possibly wrapping) stretch contains a distance along the lap."""
    assert isinstance(stretch, Stretch)
    if stretch.start <= stretch.end:
        return stretch.start <= arc_length <= stretch.end
    return arc_length >= stretch.start or arc_length <= stretch.end or arc_length >= length
