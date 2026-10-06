"""Track validation: find what's wrong with a track, and where, without changing it.

`validate` checks the control points and widths a user placed and returns a list of issues,
each with a severity, a stable code, a plain-language message, and a location the editor can
highlight. An empty list means the track is fine. Validation never raises for bad tracks and
never modifies them: reshaping a bend automatically would move the road away from the points
the user placed, so the user decides how to fix it (see ticket #12).

The editor runs this on every edit, so it is built to be fast: about 30 ms of crossing checks
on a 1.5 km track, thanks to the broad phase in `mlracecar.core.geometry`.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import (
    BoolArray,
    FloatArray,
    IntArray,
    cross,
    norm,
    polyline_crossings,
    self_intersections,
)
from mlracecar.core.track.model import Track
from mlracecar.core.track.spline import DEFAULT_SPACING


class Severity(StrEnum):
    """How serious an issue is."""

    ERROR = "error"
    """The simulator can't use the track as it is."""
    WARNING = "warning"
    """The track works, but probably not as intended."""


class IssueCode(StrEnum):
    """Stable identifiers for each kind of issue, for code that reacts to specific problems."""

    TOO_FEW_POINTS = "too-few-points"
    POINT_NOT_FINITE = "point-not-finite"
    POINTS_COINCIDE = "points-coincide"
    WIDTH_NOT_POSITIVE = "width-not-positive"
    TOO_NARROW = "too-narrow"
    TOO_SHORT = "too-short"
    EDGE_FOLDS = "edge-folds"
    SHARP_INSIDE_CORNER = "sharp-inside-corner"
    TIGHT_BEND = "tight-bend"
    TRACK_CROSSES_ITSELF = "track-crosses-itself"
    TRACK_OVERLAPS = "track-overlaps"


class ControlPointAt(NamedTuple):
    """An issue at one control point."""

    point_index: int
    """Index of the control point (0 is the first point, at the start/finish line)."""


class Stretch(NamedTuple):
    """An issue along a stretch of road, given as distances along the lap in metres.

    ``end < start`` means the stretch runs through the start/finish line.
    """

    start: float
    end: float


class Spot(NamedTuple):
    """An issue at one place on the map, in metres."""

    x: float
    y: float


type Location = ControlPointAt | Stretch | Spot | None
"""Where an issue is; ``None`` means the whole track."""


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a track."""

    severity: Severity
    code: IssueCode
    message: str
    """A plain-language explanation, ready to show to the user."""
    location: Location


@dataclass(frozen=True)
class ValidationRules:
    """Thresholds used by `validate`."""

    min_control_points: int = 4
    """Fewer points than this is an error (3 is the bare minimum to form a loop)."""
    min_width: float = 6.0
    """Narrowest allowed road, in metres: room for two 2 m cars side by side with margins."""
    min_length: float = 100.0
    """Shortest allowed lap, in metres."""
    min_inside_radius: float = 1.0
    """Bends whose inside edge curves tighter than this radius (metres) get a warning: the edge
    comes to a near-sharp point, which looks like a fold and makes an unrealistic, spiky wall."""
    min_drivable_radius: float = 6.0
    """Bends tighter than this radius (metres) get a warning: a car can't steer that sharply.
    A placeholder until vehicle parameters exist (#18)."""
    overlap_separation: float = 4.0
    """Edge crossings count as overlapping road only between parts of the track at least this
    many road widths apart along the lap. Closer crossings come from a bend folding the edge,
    which the edge-fold check already reports."""


DEFAULT_RULES = ValidationRules()


def has_errors(issues: list[ValidationIssue]) -> bool:
    """Whether any issue is an error (as opposed to only warnings)."""
    return any(issue.severity is Severity.ERROR for issue in issues)


def validate(
    control_points: ArrayLike,
    widths: ArrayLike,
    rules: ValidationRules = DEFAULT_RULES,
    *,
    spacing: float = DEFAULT_SPACING,
) -> list[ValidationIssue]:
    """Check a track and return every issue found; an empty list means it's fine.

    Args:
        control_points: Points in driving order, shape ``(P, 2)``.
        widths: Road width at each control point in metres, shape ``(P,)``.
        rules: Thresholds to check against.
        spacing: Centerline sample spacing used for the checks, in metres.

    Returns:
        Issues, ordered from input problems to whole-track problems.

    Raises:
        ValueError: If the arrays have the wrong shapes. That's a programming error, not a
            problem with the track.
    """
    points = np.asarray(control_points, dtype=np.float64)
    width = np.asarray(widths, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or width.shape != (len(points),):
        raise ValueError(
            f"need points of shape (P, 2) and widths of shape (P,), got {points.shape} "
            f"and {width.shape}"
        )

    issues, can_build = _check_inputs(points, width, rules)
    if not can_build:
        return issues
    # Degenerate curves (e.g. a cusp where the curve stops and turns back) produce inf/nan
    # values; the checks below treat those as infinitely tight bends rather than crashing.
    with np.errstate(all="ignore"):
        track = Track.build(points, width, spacing=spacing)
        issues += _check_size(track, rules)
        issues += _check_bends(track, rules)
        crossings = _check_crossings(track)
        issues += crossings
        if not crossings:  # a track that crosses itself always overlaps too; report it once
            issues += _check_overlaps(track, rules)
    return issues


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def _check_inputs(
    points: FloatArray, widths: FloatArray, rules: ValidationRules
) -> tuple[list[ValidationIssue], bool]:
    """Problems with the raw input; the bool says whether a track can still be built."""
    issues: list[ValidationIssue] = []
    if len(points) < rules.min_control_points:
        issues.append(
            _error(
                IssueCode.TOO_FEW_POINTS,
                f"A track needs at least {rules.min_control_points} points; this one has "
                f"{len(points)}.",
                None,
            )
        )
    for index in np.flatnonzero(~np.all(np.isfinite(points), axis=1)):
        issues.append(
            _error(
                IssueCode.POINT_NOT_FINITE,
                f"Point {index} has an invalid position.",
                ControlPointAt(int(index)),
            )
        )
    for index in np.flatnonzero(~(widths > 0)):
        issues.append(
            _error(
                IssueCode.WIDTH_NOT_POSITIVE,
                f"Point {index} has a width of {widths[index]:g} m; widths must be positive.",
                ControlPointAt(int(index)),
            )
        )
    if len(points) >= 2 and np.all(np.isfinite(points)):
        gaps = norm(np.roll(points, -1, axis=0) - points)
        for index in np.flatnonzero(gaps == 0):
            following = (index + 1) % len(points)
            issues.append(
                _error(
                    IssueCode.POINTS_COINCIDE,
                    f"Point {index} is on top of point {following}; move or delete one of them.",
                    ControlPointAt(int(index)),
                )
            )
    blocking = {IssueCode.POINT_NOT_FINITE, IssueCode.WIDTH_NOT_POSITIVE, IssueCode.POINTS_COINCIDE}
    can_build = len(points) >= 3 and not any(issue.code in blocking for issue in issues)
    return issues, can_build


def _check_size(track: Track, rules: ValidationRules) -> list[ValidationIssue]:
    """Lap too short, or road narrower than allowed at a control point.

    Checking the control points is enough: widths between them are blended without ever
    going below the narrower of the two.
    """
    issues: list[ValidationIssue] = []
    if track.length < rules.min_length:
        issues.append(
            _error(
                IssueCode.TOO_SHORT,
                f"The lap is {track.length:.0f} m long; the minimum is {rules.min_length:.0f} m.",
                None,
            )
        )
    for index in np.flatnonzero(track.control_widths < rules.min_width):
        issues.append(
            _error(
                IssueCode.TOO_NARROW,
                f"The road at point {index} is {track.control_widths[index]:.1f} m wide; the "
                f"minimum is {rules.min_width:.1f} m.",
                ControlPointAt(int(index)),
            )
        )
    return issues


def _check_bends(track: Track, rules: ValidationRules) -> list[ValidationIssue]:
    """Bends too tight for the road width or for a car to steer.

    A *bend* here is a continuous stretch where the road is too tight by any measure. Each bend
    is reported once, by its most serious problem: an edge-fold error if the inside edge folds
    anywhere in it; otherwise a sharp-inside-corner warning if the inside edge comes to a
    near-point; otherwise a tight-bend warning.
    """
    curvature = np.nan_to_num(np.abs(track.centerline.curvature), nan=np.inf)
    radius = 1 / curvature
    inside_radius = radius - track.width / 2  # radius of the inside edge; below 0 it folds
    folds = inside_radius < 0
    sharp = inside_radius < rules.min_inside_radius
    problem = folds | sharp | (radius < rules.min_drivable_radius)

    issues: list[ValidationIssue] = []
    for samples in _runs(problem):
        tightest = samples[np.argmax(curvature[samples])]
        start, end = _stretch(track, samples)
        where = f"The bend from {np.floor(start):.0f} m to {np.ceil(end):.0f} m"
        if folds[samples].any():
            folding = samples[folds[samples]]
            worst = folding[np.argmax(curvature[folding] * track.width[folding])]
            side = "left" if track.centerline.curvature[worst] > 0 else "right"
            message = (
                f"{where} is too tight for the road: its radius is {radius[worst]:.1f} m but "
                f"the road is {track.width[worst] / 2:.1f} m wide on each side, so the {side} "
                "edge folds over itself. Widen the bend or narrow the road there."
            )
            issues.append(_error(IssueCode.EDGE_FOLDS, message, Stretch(start, end)))
        elif sharp[samples].any():
            cornered = samples[sharp[samples]]
            worst = cornered[np.argmin(inside_radius[cornered])]
            side = "left" if track.centerline.curvature[worst] > 0 else "right"
            message = (
                f"{where} makes the {side} edge come to a sharp point: its inside radius is only "
                f"{inside_radius[worst]:.1f} m (at least {rules.min_inside_radius:.1f} m "
                "recommended). Widen the bend or narrow the road there."
            )
            issues.append(
                ValidationIssue(
                    Severity.WARNING, IssueCode.SHARP_INSIDE_CORNER, message, Stretch(start, end)
                )
            )
        else:
            message = (
                f"{where} is tighter than a car can steer: radius {radius[tightest]:.1f} m, "
                f"minimum {rules.min_drivable_radius:.1f} m."
            )
            issues.append(
                ValidationIssue(
                    Severity.WARNING, IssueCode.TIGHT_BEND, message, Stretch(start, end)
                )
            )
    return issues


def _check_crossings(track: Track) -> list[ValidationIssue]:
    """The centerline crossing itself, e.g. a loop from the spline overshooting uneven points."""
    centerline = track.centerline.points
    pairs = self_intersections(centerline, closed=True)
    spots = _intersection_points(centerline, centerline, pairs)
    issues: list[ValidationIssue] = []
    for group in _cluster(spots, radius=float(track.width.max())):
        first, second = sorted(track.centerline.arc_length[pairs[group[0]]])
        x, y = spots[group].mean(axis=0)
        issues.append(
            _error(
                IssueCode.TRACK_CROSSES_ITSELF,
                f"The track crosses itself here, at {first:.0f} m and {second:.0f} m along the lap."
                " Spread the nearby points more evenly or move them apart.",
                Spot(float(x), float(y)),
            )
        )
    return issues


def _check_overlaps(track: Track, rules: ValidationRules) -> list[ValidationIssue]:
    """Distant parts of the road touching each other: their edges cross."""
    arc = track.centerline.arc_length
    min_apart = rules.overlap_separation * float(track.width.max())
    spots: list[FloatArray] = []
    places: list[tuple[float, float]] = []
    for first, second, pairs in (
        (track.left, track.left, self_intersections(track.left, closed=True)),
        (track.right, track.right, self_intersections(track.right, closed=True)),
        (track.left, track.right, polyline_crossings(track.left, track.right, closed=True)),
    ):
        apart = np.abs(arc[pairs[:, 0]] - arc[pairs[:, 1]])
        distant = pairs[np.minimum(apart, track.length - apart) >= min_apart]
        spots.append(_intersection_points(first, second, distant))
        places += [(float(arc[i]), float(arc[j])) for i, j in distant]
    all_spots = np.concatenate(spots)

    issues: list[ValidationIssue] = []
    # Where two road parts overlap, their edges cross where the overlap begins and ends, a
    # road-width or so apart; grouping within two widths reports each overlap once.
    for group in _cluster(all_spots, radius=2 * float(track.width.max())):
        first, second = np.mean([sorted(places[k]) for k in group], axis=0)
        x, y = all_spots[group].mean(axis=0)
        issues.append(
            _error(
                IssueCode.TRACK_OVERLAPS,
                f"Two parts of the road overlap here, at about {first:.0f} m and {second:.0f} m "
                "along the lap. Move them apart or make the road narrower there.",
                Spot(float(x), float(y)),
            )
        )
    return issues


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _error(code: IssueCode, message: str, location: Location) -> ValidationIssue:
    return ValidationIssue(Severity.ERROR, code, message, location)


def _runs(mask: BoolArray) -> list[IntArray]:
    """Sample indices of each run of ``True`` in a cyclic mask (a run may wrap around)."""
    count = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [np.arange(count)]
    shift = int(np.argmin(mask))  # start scanning at a False so no run is split
    edges = np.diff(np.concatenate(([0], np.roll(mask, -shift).astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(np.arange(a, b) + shift) % count for a, b in zip(starts, stops, strict=True)]


def _stretch(track: Track, samples: IntArray) -> Stretch:
    """The stretch of road covered by a run of samples (each sample covers one step ahead)."""
    step = track.length / len(track.centerline.arc_length)
    start = float(track.centerline.arc_length[samples[0]])
    end = float((track.centerline.arc_length[samples[-1]] + step) % track.length)
    return Stretch(start, end)


def _intersection_points(first: FloatArray, second: FloatArray, pairs: IntArray) -> FloatArray:
    """Where segment ``i`` of closed polyline ``first`` meets segment ``j`` of ``second``."""
    i, j = pairs[:, 0], pairs[:, 1]
    p, r = first[i], np.roll(first, -1, axis=0)[i] - first[i]
    q, s = second[j], np.roll(second, -1, axis=0)[j] - second[j]
    along = cross(q - p, s) / cross(r, s)
    result: FloatArray = p + along[:, None] * r
    return result


def _cluster(spots: FloatArray, radius: float) -> list[IntArray]:
    """Group spots lying within ``radius`` of each group's first spot (one group per place)."""
    remaining = np.arange(len(spots))
    groups: list[IntArray] = []
    while len(remaining):
        near = norm(spots[remaining] - spots[remaining[0]]) <= radius
        groups.append(remaining[near])
        remaining = remaining[~near]
    return groups
