"""The track editor's model: a track being drawn, and the edits a user can make to it.

A `TrackDraft` is immutable: every edit returns a new draft and leaves the old one untouched
(ADR-0011). That makes undo/redo a list of drafts, lets the view compare before and after, and
means each draft can cache its derived track and its validation issues.

Nothing here touches pygame or the screen; positions are world coordinates in metres. The view
converts mouse positions before calling in, so all of the editing logic is testable headless.
"""

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Self

import numpy as np
from pydantic import JsonValue

from mlracecar.core.geometry import FloatArray, norm, project_onto_polyline
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import ValidationIssue, has_errors, validate
from mlracecar.editor import corners
from mlracecar.editor.corners import CornerLimits
from mlracecar.io.track_file import TrackFile

type Point = tuple[float, float]

DEFAULT_NAME = "Untitled track"
"""Name of a new track until the user names it (or saves it, which names it after the file)."""

DEFAULT_WIDTH = 12.0
"""Road width given to the first points of a new track, in metres."""

MIN_WIDTH = 1.0
"""Narrowest width the editor lets a point have. The track checks flag anything under 6 m."""

MAX_WIDTH = 100.0
"""Widest width the editor lets a point have, in metres."""


@dataclass(frozen=True)
class TrackDraft:
    """A track as it's being edited. Every edit method returns a new draft."""

    points: tuple[Point, ...] = ()
    """Control points in driving order, in metres. Point 0 is on the start/finish line."""
    widths: tuple[float, ...] = ()
    """Road width at each point, in metres."""
    name: str = DEFAULT_NAME
    author: str = ""
    description: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict, hash=False, repr=False)
    """Free-form data from the track file, kept as it was (an editor's view settings, say)."""

    def __post_init__(self) -> None:
        if len(self.points) != len(self.widths):
            raise ValueError(f"{len(self.points)} points but {len(self.widths)} widths")

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #

    @classmethod
    def from_track_file(cls, track_file: TrackFile) -> Self:
        """Start editing a loaded track file."""
        return cls(
            points=tuple((p.x, p.y) for p in track_file.control_points),
            widths=tuple(p.width for p in track_file.control_points),
            name=track_file.name,
            author=track_file.author,
            description=track_file.description,
            metadata=dict(track_file.metadata),
        )

    def to_track_file(self) -> TrackFile:
        """The draft as a track file, ready to save.

        Drafts with track errors can be saved (they're work in progress), but a file needs at
        least 3 points.

        Raises:
            mlracecar.io.track_file.TrackFileError: If the draft can't be a track file yet.
        """
        return TrackFile.from_arrays(
            self.name,
            np.array(self.points, dtype=np.float64).reshape(-1, 2),
            self.widths,
            author=self.author,
            description=self.description,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------------ #
    # Derived results (computed once per draft)
    # ------------------------------------------------------------------ #

    @cached_property
    def issues(self) -> list[ValidationIssue]:
        """Everything the track checks find wrong with this draft."""
        return validate(np.array(self.points).reshape(-1, 2), np.array(self.widths))

    @cached_property
    def track(self) -> Track | None:
        """The full track, or ``None`` while the points can't form one (see `issues` for why)."""
        try:
            return Track.build(self.points, self.widths)
        except ValueError:
            return None

    @property
    def is_raceable(self) -> bool:
        """Whether the track has no errors (warnings are fine)."""
        return self.track is not None and not has_errors(self.issues)

    # ------------------------------------------------------------------ #
    # Finding things under the cursor
    # ------------------------------------------------------------------ #

    def point_near(self, position: Point, radius: float) -> int | None:
        """The point closest to ``position`` if it's within ``radius`` metres, else ``None``."""
        if not self.points:
            return None
        distances = norm(np.array(self.points) - np.asarray(position))
        index = int(np.argmin(distances))
        return index if distances[index] <= radius else None

    def is_on_road(self, position: Point) -> bool:
        """Whether ``position`` is on the road (never, while the points can't form a track)."""
        track = self.track
        if track is None:
            return False
        nearest = project_onto_polyline([position], track.centerline.points, closed=True)
        return bool(abs(nearest.offset[0]) <= track.width[int(nearest.segment[0])] / 2)

    # ------------------------------------------------------------------ #
    # Edits
    # ------------------------------------------------------------------ #

    def append_point(self, position: Point) -> Self:
        """Add a point after the last one, extending the track (the loop closes back to point 0).

        This is what a click off the road does: clicking around in order draws a track. The new
        point gets the last point's width.
        """
        width = self.widths[-1] if self.widths else DEFAULT_WIDTH
        return self._with((*self.points, position), (*self.widths, width))

    def insert_point(self, position: Point) -> Self:
        """Insert a point into the stretch of road nearest ``position``.

        This is what a click on the road does, to refine a track: the point goes between the two
        points around that stretch, with the road width the track has there, so the shape barely
        changes until the new point is moved. With fewer than 3 points there are no stretches
        yet, so it appends instead.
        """
        if len(self.points) < 3:
            return self.append_point(position)
        after, width = self._nearest_stretch(position)
        return self._with(
            (*self.points[: after + 1], position, *self.points[after + 1 :]),
            (*self.widths[: after + 1], width, *self.widths[after + 1 :]),
        )

    def move_point(self, index: int, position: Point) -> Self:
        """Move a point."""
        self._check(index)
        return self._with(_replaced(self.points, index, position), self.widths)

    def delete_point(self, index: int) -> Self:
        """Remove a point. The next point takes its index; deleting point 0 moves the start."""
        self._check(index)
        return self._with(
            self.points[:index] + self.points[index + 1 :],
            self.widths[:index] + self.widths[index + 1 :],
        )

    def set_width(self, index: int, width: float) -> Self:
        """Set the road width at a point, kept between `MIN_WIDTH` and `MAX_WIDTH`."""
        self._check(index)
        clamped = float(np.clip(width, MIN_WIDTH, MAX_WIDTH))
        return self._with(self.points, _replaced(self.widths, index, clamped))

    def change_width(self, index: int, by: float) -> Self:
        """Widen (positive ``by``) or narrow the road at a point, in metres."""
        self._check(index)
        return self.set_width(index, self.widths[index] + by)

    def reverse(self) -> Self:
        """Drive the track the other way round, keeping point 0 on the start/finish line."""
        order = [0, *range(len(self.points) - 1, 0, -1)] if self.points else []
        return self._with(
            tuple(self.points[i] for i in order), tuple(self.widths[i] for i in order)
        )

    def set_start(self, index: int) -> Self:
        """Make a point the start/finish line, keeping the driving direction."""
        self._check(index)
        return self._with(
            self.points[index:] + self.points[:index], self.widths[index:] + self.widths[:index]
        )

    def corner_limits(self, index: int) -> CornerLimits:
        """The radii the corner at a point can be rounded to (see `round_corner`).

        Raises:
            mlracecar.editor.corners.CornerError: If the point isn't a corner, or no radius
                fits there.
        """
        self._check(index)
        return corners.corner_limits(self._point_array(), np.array(self.widths), index)

    def round_corner(self, index: int, radius: float) -> tuple[Self, int]:
        """Replace a sharp point with points along a smooth bend of ``radius`` metres.

        The bend eases into a circular arc of that radius and out again, touching the straights
        to the neighbouring corners, whose points are replaced too (see
        `mlracecar.editor.corners`). Returns the new draft and the index of the bend's middle
        point.

        Raises:
            mlracecar.editor.corners.CornerError: If the point isn't a corner, or ``radius``
                doesn't fit (`corner_limits` gives the radii that do).
        """
        self._check(index)
        rounded = corners.round_corner(self._point_array(), np.array(self.widths), index, radius)
        draft = self._with(
            tuple((float(x), float(y)) for x, y in rounded.points),
            tuple(float(width) for width in rounded.widths),
        )
        return draft, rounded.middle

    def rename(self, name: str) -> Self:
        """Change the track's name (an empty name keeps the old one)."""
        return replace(self, name=name.strip() or self.name)

    def rounded(self, decimals: int) -> Self:
        """Round every position and width to ``decimals`` decimal places (3 is a millimetre)."""

        def tidy(value: float) -> float:
            return round(value, decimals) + 0.0  # + 0.0 turns -0.0 into 0.0

        return self._with(
            tuple((tidy(x), tidy(y)) for x, y in self.points),
            tuple(tidy(width) for width in self.widths),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _with(self, points: tuple[Point, ...], widths: tuple[float, ...]) -> Self:
        return replace(self, points=points, widths=widths)

    def _point_array(self) -> FloatArray:
        return np.array(self.points, dtype=np.float64).reshape(-1, 2)

    def _check(self, index: int) -> None:
        if not 0 <= index < len(self.points):
            raise IndexError(f"no point {index}; the draft has {len(self.points)} points")

    def _nearest_stretch(self, position: Point) -> tuple[int, float]:
        """The point after which a click belongs, and the road width at that spot.

        Uses the drawn curve when there is one, since it's what the user sees and can bulge
        away from the straight lines between points; otherwise the straight lines.
        """
        track = self.track
        if track is not None:
            nearest = project_onto_polyline([position], track.centerline.points, closed=True)
            sample = int(nearest.segment[0])
            return int(track.centerline.piece[sample]), float(track.width[sample])
        outline: FloatArray = np.array(self.points)
        segment = int(project_onto_polyline([position], outline, closed=True).segment[0])
        following = (segment + 1) % len(self.widths)
        return segment, (self.widths[segment] + self.widths[following]) / 2


def _replaced[T](values: tuple[T, ...], index: int, value: T) -> tuple[T, ...]:
    return (*values[:index], value, *values[index + 1 :])
