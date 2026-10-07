"""The track model: everything the simulation needs, derived from control points and widths.

A track is stored as control points with a road width at each (ADR-0004). From those, this
module derives the smooth centerline (`spline`), the width at every sample, the left and
right road edges, the start/finish line and checkpoints, and the starting grid.

Conventions: distances along the track (``arc_length``) start at control point 0, which is
also where the start/finish line is. "Left" means left of the driving direction, so the left
edge is the inside of a counter-clockwise track and the outside of a clockwise one.
"""

from dataclasses import dataclass
from typing import NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray, norm
from mlracecar.core.track.spline import DEFAULT_SPACING, Centerline, ClosedSpline

DEFAULT_CHECKPOINT_SPACING = 20.0
"""Default distance between checkpoints along the track, in metres."""

MIN_CHECKPOINTS = 3
"""A lap needs at least this many checkpoints so that driving backwards can be detected."""


class Pose(NamedTuple):
    """Positions and headings, one per entry."""

    position: FloatArray
    """Positions, shape ``(..., 2)``."""
    heading: FloatArray
    """Headings in radians, counter-clockwise from +x, shape ``(...)``."""


class Checkpoints(NamedTuple):
    """Lines across the road at evenly spaced distances. Checkpoint 0 is the start/finish line."""

    arc_length: FloatArray
    """Distance along the track of each checkpoint, shape ``(M,)``; the first is 0."""
    left: FloatArray
    """End of each line on the left edge, shape ``(M, 2)``."""
    right: FloatArray
    """End of each line on the right edge, shape ``(M, 2)``."""


class GridLayout(NamedTuple):
    """Spacing of the starting grid. The default car size is the default car's; `World` passes
    the size of the car it simulates."""

    car_length: float = 4.5
    """Length of a car in metres."""
    car_width: float = 2.0
    """Width of a car in metres."""
    gap: float = 1.5
    """Clear space between the back of one car and the front of the next, in metres."""
    first_gap: float = 1.0
    """Distance from the start line back to the front of the pole-position car, in metres."""


DEFAULT_GRID_LAYOUT = GridLayout()


@dataclass(frozen=True, eq=False, repr=False)
class Track:
    """An immutable track with all derived geometry. Create one with :meth:`build`."""

    spline: ClosedSpline
    """The smooth closed centerline through the control points."""
    control_widths: FloatArray
    """Road width at each control point in metres, shape ``(P,)``."""
    centerline: Centerline
    """The centerline sampled at even spacing (``n`` samples)."""
    width: FloatArray
    """Road width at each centerline sample, shape ``(n,)``."""
    left: FloatArray
    """Left road edge, one point per sample, shape ``(n, 2)``. A closed polyline."""
    right: FloatArray
    """Right road edge, one point per sample, shape ``(n, 2)``. A closed polyline."""
    checkpoints: Checkpoints
    """Start/finish line and checkpoints, evenly spaced along the track."""

    @classmethod
    def build(
        cls,
        control_points: ArrayLike,
        widths: ArrayLike,
        *,
        spacing: float = DEFAULT_SPACING,
        checkpoint_spacing: float = DEFAULT_CHECKPOINT_SPACING,
    ) -> Self:
        """Derive a track from its control points and the road width at each.

        Args:
            control_points: Points in driving order, shape ``(P, 2)``, ``P >= 3``.
            widths: Road width at each control point in metres, shape ``(P,)``, all positive.
            spacing: Target distance between centerline samples in metres.
            checkpoint_spacing: Target distance between checkpoints in metres. The actual
                spacing is adjusted so checkpoints are evenly spread over the lap, with at
                least `MIN_CHECKPOINTS` of them.

        Returns:
            The track.

        Raises:
            ValueError: If the widths don't match the points or aren't positive, if the
                checkpoint spacing isn't positive, or if the spline can't be built.
        """
        spline = ClosedSpline.through(control_points)
        control_widths = np.asarray(widths, dtype=np.float64)
        if control_widths.shape != (len(spline.control_points),):
            raise ValueError(
                f"need one width per control point ({len(spline.control_points)}), "
                f"got shape {control_widths.shape}"
            )
        if not np.all(control_widths > 0):
            raise ValueError("widths must be positive")
        if not checkpoint_spacing > 0:
            raise ValueError(f"checkpoint_spacing must be positive, got {checkpoint_spacing}")

        centerline = spline.resample(spacing)
        width = _blend_widths(control_widths, centerline.piece, centerline.fraction)
        half = (width / 2)[:, None] * centerline.normal

        count = max(MIN_CHECKPOINTS, round(spline.length / checkpoint_spacing))
        arc_length = np.arange(count, dtype=np.float64) * (spline.length / count)
        half_width = _width_at(spline, control_widths, arc_length) / 2
        checkpoints = Checkpoints(
            arc_length,
            _pose_at(spline, arc_length, half_width).position,
            _pose_at(spline, arc_length, -half_width).position,
        )
        return cls(
            spline=spline,
            control_widths=control_widths,
            centerline=centerline,
            width=width,
            left=centerline.points + half,
            right=centerline.points - half,
            checkpoints=checkpoints,
        )

    @property
    def length(self) -> float:
        """Length of the lap along the centerline, in metres."""
        return self.spline.length

    def width_at(self, arc_length: ArrayLike) -> FloatArray:
        """Road width at any distance along the track (wrapping around)."""
        return _width_at(self.spline, self.control_widths, arc_length)

    def pose_at(self, arc_length: ArrayLike, lateral_offset: ArrayLike = 0.0) -> Pose:
        """Position and driving direction at any distance along the track.

        Args:
            arc_length: Distances along the track in metres; values wrap around the lap.
            lateral_offset: Sideways shift from the centerline in metres, positive to the
                left; broadcasts against ``arc_length``.

        Returns:
            Positions shifted sideways from the centerline, facing the driving direction.
        """
        return _pose_at(self.spline, arc_length, lateral_offset)

    def start_grid(self, count: int, layout: GridLayout = DEFAULT_GRID_LAYOUT) -> Pose:
        """Starting positions for ``count`` cars: a staggered grid, two cars wide.

        Car 0 (pole position) starts on the left, just behind the start line; the cars then
        alternate right, left, right... Each car is one car length plus ``layout.gap`` further
        back than the previous one, so cars never overlap along the track, and each faces the
        driving direction at its spot. Lanes sit a quarter of the road width either side of
        the centerline, moved inwards if needed to keep the whole car on the road.

        Args:
            count: Number of cars, zero or more.
            layout: Car size and spacing.

        Returns:
            Poses of shape ``(count, 2)`` and ``(count,)``.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"count must not be negative, got {count}")
        behind_line = (
            layout.first_gap
            + layout.car_length / 2
            + np.arange(count) * (layout.car_length + layout.gap)
        )
        arc_length = -behind_line  # wraps to just before the end of the lap
        width = self.width_at(arc_length)
        lane_offset = np.clip(np.minimum(width / 4, (width - layout.car_width) / 2), 0.0, None)
        side = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)  # left, right, left, ...
        return self.pose_at(arc_length, side * lane_offset)

    def __repr__(self) -> str:
        return (
            f"Track(length={self.length:.1f} m, {len(self.control_widths)} control points, "
            f"{len(self.width)} samples, {len(self.checkpoints.arc_length)} checkpoints)"
        )


def _pose_at(spline: ClosedSpline, arc_length: ArrayLike, lateral_offset: ArrayLike) -> Pose:
    """Position and heading at distances along the spline, shifted sideways (left = +)."""
    piece, fraction = spline.locate(arc_length)
    velocity = spline.velocity(piece, fraction)
    tangent = velocity / norm(velocity)[..., None]
    normal = np.stack([-tangent[..., 1], tangent[..., 0]], axis=-1)
    offset = np.asarray(lateral_offset, dtype=np.float64)[..., None]
    position = spline.position(piece, fraction) + offset * normal
    heading: FloatArray = np.arctan2(tangent[..., 1], tangent[..., 0])
    return Pose(position, heading)


def _width_at(spline: ClosedSpline, widths: FloatArray, arc_length: ArrayLike) -> FloatArray:
    """Road width at distances along the spline."""
    piece, fraction = spline.locate(arc_length)
    return _blend_widths(widths, piece, fraction)


def _blend_widths(widths: FloatArray, piece: ArrayLike, fraction: ArrayLike) -> FloatArray:
    """Width between control points, blended with a smoothstep curve.

    The blend is flat at each control point (so the edges have no kinks there) and never goes
    outside the range of its two control widths (so the road is never narrower than the
    narrowest width the user set).
    """
    start = np.asarray(piece, dtype=np.intp)
    f = np.asarray(fraction, dtype=np.float64)
    smooth = f * f * (3 - 2 * f)
    result: FloatArray = (
        widths[start] + (widths[(start + 1) % len(widths)] - widths[start]) * smooth
    )
    return result
