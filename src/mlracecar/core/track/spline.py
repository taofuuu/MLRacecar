"""Closed C2 cubic splines: turning clicked control points into a smooth loop.

A track is stored as a few control points (ADR-0004). This module draws the smooth closed
curve through them and resamples it at even spacing, giving the simulator a dense centerline
with a direction and curvature at every sample.

Why a *C2* cubic spline (ADR-0010): the curve passes exactly through every control point, and
its position, direction, **and curvature** all change continuously. A corner drawn as a steady
arc therefore bends steadily, which matters for how the car drives and for the curvature the
agent observes. Knots are spaced by the square root of the distance between neighbouring
points ("centripetal" spacing), which keeps the curve close to the control polygon.

Each piece between two neighbouring control points is a cubic polynomial in a local
parameter ``fraction`` from 0 to 1. Direction and curvature therefore come from exact
derivatives, not numerical estimates.
"""

from dataclasses import dataclass
from typing import NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray, IntArray, cross, norm

DEFAULT_SPACING = 0.5
"""Default distance between resampled centerline points, in metres."""

ALPHA = 0.5
"""Knot exponent: the knot interval of a piece is ``distance ** ALPHA`` (0.5 = centripetal)."""

_TABLE_STEPS_PER_PIECE = 512
"""Sub-steps per piece in the arc-length lookup table. Resampled points are found by linear
interpolation in this table; 512 keeps the along-road spacing error below 0.03% (64 allowed
about 1.2%) for about 1 ms more per track."""


class Centerline(NamedTuple):
    """A spline resampled at even spacing. Array fields have one row per sample."""

    points: FloatArray
    """Sample positions, shape ``(n, 2)``. The first sample is control point 0."""
    arc_length: FloatArray
    """Distance along the loop from the first sample, shape ``(n,)``. Starts at 0."""
    tangent: FloatArray
    """Unit direction of travel, shape ``(n, 2)``."""
    normal: FloatArray
    """Unit vector pointing left of the direction of travel, shape ``(n, 2)``."""
    curvature: FloatArray
    """Signed curvature in 1/m: positive when turning left, 1/radius in magnitude, ``(n,)``."""
    piece: IntArray
    """Index of the spline piece (control point ``i`` to ``i + 1``) each sample is on, ``(n,)``."""
    fraction: FloatArray
    """Position within that piece, from 0 (at control point ``i``) towards 1, shape ``(n,)``."""
    length: float
    """Total length of the loop in metres."""


@dataclass(frozen=True, eq=False)
class ClosedSpline:
    """A closed C2 cubic spline through control points. Create one with :meth:`through`.

    Piece ``i`` runs from control point ``i`` to control point ``i + 1`` (the last piece
    returns to point 0). Methods take a piece index and a ``fraction`` in ``[0, 1]`` and
    broadcast over both.
    """

    control_points: FloatArray
    """Control points in order of travel, shape ``(P, 2)``."""
    coefficients: FloatArray
    """Cubic coefficients of each piece, ``a f^3 + b f^2 + c f + d``, shape ``(P, 4, 2)``."""
    arc_table: FloatArray
    """Cumulative length at evenly spaced parameters, used to find positions by distance."""

    @classmethod
    def through(cls, control_points: ArrayLike) -> Self:
        """Fit the spline through control points.

        Args:
            control_points: At least 3 points, shape ``(P, 2)``, in order of travel. Neighbouring
                points (including the last and the first) must differ.

        Returns:
            The fitted spline.

        Raises:
            ValueError: If there are fewer than 3 points or two neighbouring points coincide.
        """
        points = np.asarray(control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"need at least 3 control points of shape (P, 2), got {points.shape}")
        gaps = norm(np.roll(points, -1, axis=0) - points)
        if np.any(gaps == 0):
            duplicates = np.flatnonzero(gaps == 0).tolist()
            raise ValueError(f"control points coincide with the next one at indices {duplicates}")

        coefficients = _piece_coefficients(points)
        steps = np.linspace(0.0, 1.0, _TABLE_STEPS_PER_PIECE, endpoint=False)
        pieces = np.repeat(np.arange(len(points)), _TABLE_STEPS_PER_PIECE)
        samples = _evaluate(coefficients, pieces, np.tile(steps, len(points)), derivative=0)
        samples = np.vstack([samples, points[:1]])  # close the loop back at point 0
        arc_table = np.concatenate(([0.0], np.cumsum(norm(np.diff(samples, axis=0)))))
        return cls(points, coefficients, arc_table)

    @property
    def length(self) -> float:
        """Total length of the closed loop in metres."""
        return float(self.arc_table[-1])

    def position(self, piece: ArrayLike, fraction: ArrayLike) -> FloatArray:
        """Point on the spline, shape ``(..., 2)``."""
        return _evaluate(self.coefficients, piece, fraction, derivative=0)

    def velocity(self, piece: ArrayLike, fraction: ArrayLike) -> FloatArray:
        """First derivative with respect to ``fraction``: points along the direction of travel."""
        return _evaluate(self.coefficients, piece, fraction, derivative=1)

    def curvature(self, piece: ArrayLike, fraction: ArrayLike) -> FloatArray:
        """Signed curvature in 1/m: positive when turning left (counter-clockwise)."""
        first = self.velocity(piece, fraction)
        second = _evaluate(self.coefficients, piece, fraction, derivative=2)
        result: FloatArray = cross(first, second) / norm(first) ** 3
        return result

    def locate(self, arc_length: ArrayLike) -> tuple[IntArray, FloatArray]:
        """Piece and fraction at given distances along the loop.

        Distances wrap around: ``-1`` is one metre before the start, ``length + 1`` one metre
        after it.

        Args:
            arc_length: Distances along the loop from control point 0, in metres, any shape.

        Returns:
            ``(piece, fraction)`` arrays with the same shape as ``arc_length``.
        """
        distance = np.mod(np.asarray(arc_length, dtype=np.float64), self.length)
        # Invert the arc-length table: distance along the loop -> global parameter piece+fraction.
        parameter_grid = np.linspace(0.0, len(self.control_points), len(self.arc_table))
        parameter = np.asarray(np.interp(distance, self.arc_table, parameter_grid))
        piece = np.minimum(parameter.astype(np.intp), len(self.control_points) - 1)
        fraction: FloatArray = parameter - piece
        return piece, fraction

    def arc_length_at(self, piece: ArrayLike, fraction: ArrayLike) -> FloatArray:
        """Distance along the loop from control point 0 to the given spots (inverse of `locate`).

        Args:
            piece: Piece indices, any shape.
            fraction: Positions within each piece, from 0 to 1, broadcastable against ``piece``.

        Returns:
            Distances in metres, the broadcast shape of the inputs.
        """
        parameter = np.asarray(piece, dtype=np.float64) + np.asarray(fraction, dtype=np.float64)
        parameter_grid = np.linspace(0.0, len(self.control_points), len(self.arc_table))
        result: FloatArray = np.asarray(np.interp(parameter, parameter_grid, self.arc_table))
        return result

    def resample(self, spacing: float = DEFAULT_SPACING) -> Centerline:
        """Sample the loop at (almost exactly) even distances.

        The spacing is adjusted slightly so a whole number of samples fits the loop: with
        ``n = round(length / spacing)`` samples, they are ``length / n`` apart.

        Args:
            spacing: Target distance between samples in metres.

        Returns:
            The resampled centerline.

        Raises:
            ValueError: If ``spacing`` is not positive.
        """
        if not spacing > 0:
            raise ValueError(f"spacing must be positive, got {spacing}")
        count = max(3, round(self.length / spacing))
        arc_length = np.arange(count, dtype=np.float64) * (self.length / count)
        piece, fraction = self.locate(arc_length)

        velocity = self.velocity(piece, fraction)
        tangent = velocity / norm(velocity)[:, None]
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # tangent turned 90° left
        return Centerline(
            points=self.position(piece, fraction),
            arc_length=arc_length,
            tangent=tangent,
            normal=normal,
            curvature=self.curvature(piece, fraction),
            piece=piece,
            fraction=fraction,
            length=self.length,
        )


def _piece_coefficients(points: FloatArray) -> FloatArray:
    """Cubic coefficients of every piece of the closed C2 interpolating spline.

    With knot parameter ``t`` and knot intervals ``h_i = |p_{i+1} - p_i| ** ALPHA``, the
    second derivatives ``m_i = d²p/dt²`` at the control points satisfy the cyclic system

        h_{i-1} m_{i-1} + 2 (h_{i-1} + h_i) m_i + h_i m_{i+1}
            = 6 ((p_{i+1} - p_i) / h_i - (p_i - p_{i-1}) / h_{i-1}),

    which makes the curve's second derivative (and so its curvature) continuous. The matrix is
    strictly diagonally dominant, so the system always has a unique solution. Each piece is
    then rewritten as ``a f^3 + b f^2 + c f + d`` in its local ``fraction`` f = (t - t_i) / h_i.
    """
    count = len(points)
    following = np.roll(points, -1, axis=0)
    h = norm(following - points) ** ALPHA  # knot interval of each piece
    h_before = np.roll(h, 1)
    slope = (following - points) / h[:, None]  # chord slope of each piece in t

    rows = np.arange(count)
    system = np.zeros((count, count))
    system[rows, rows] = 2 * (h_before + h)
    system[rows, (rows - 1) % count] = h_before
    system[rows, (rows + 1) % count] = h
    second = np.linalg.solve(system, 6 * (slope - np.roll(slope, 1, axis=0)))

    m0, m1 = second, np.roll(second, -1, axis=0)
    h2 = (h**2)[:, None]
    a = h2 * (m1 - m0) / 6
    b = h2 * m0 / 2
    c = (following - points) - h2 * (2 * m0 + m1) / 6
    return np.stack([a, b, c, points], axis=1)


def _evaluate(
    coefficients: FloatArray, piece: ArrayLike, fraction: ArrayLike, *, derivative: int
) -> FloatArray:
    """Evaluate the cubic of each piece (or its first or second derivative) at ``fraction``."""
    a, b, c, d = np.moveaxis(coefficients[np.asarray(piece, dtype=np.intp)], -2, 0)
    f = np.asarray(fraction, dtype=np.float64)[..., None]
    if derivative == 0:
        result: FloatArray = ((a * f + b) * f + c) * f + d
    elif derivative == 1:
        result = (3 * a * f + 2 * b) * f + c
    else:
        result = 6 * a * f + 2 * b
    return result
