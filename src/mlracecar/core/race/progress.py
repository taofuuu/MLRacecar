"""Where cars are along the track: their projection onto the centerline.

Projecting a car onto the whole centerline compares it with every sample of the lap. Between
two updates a car only moves a few metres, so after the first time each car is compared only
with the stretch of centerline around where it was (amortized O(1) per car). That also keeps a
car's progress on its own stretch of road: cutting across the grass to a part of the track that
passes close by (the other side of a hairpin, say) doesn't move it along the lap.
"""

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray, IntArray, cross
from mlracecar.core.track.model import Track

MARGIN = 5.0
"""Metres of centerline searched beyond how far a car can move between two updates."""

_CHUNK = 256
"""Cars projected onto the whole centerline at once, to keep memory use small."""


class RoadPosition(NamedTuple):
    """Where points are relative to the centerline; one entry per point."""

    segment: IntArray
    """The centerline segment (from sample ``i`` to ``i + 1``) closest to each point."""
    arc_length: FloatArray
    """Distance along the lap, in ``[0, length)`` metres."""
    offset: FloatArray
    """Distance from the centerline in metres, positive to the left."""
    heading: FloatArray
    """The road's direction there, in radians."""
    width: FloatArray
    """The road's width there, in metres."""


class RoadLocator:
    """Finds where points are along a track.

    Args:
        track: The track.
        reach: The farthest a car can move between two updates, in metres.
    """

    def __init__(self, track: Track, reach: float) -> None:
        line = track.centerline
        self.length = line.length
        self._starts = line.points
        self._vectors = np.roll(line.points, -1, axis=0) - line.points
        self._squared_lengths = np.einsum("si,si->s", self._vectors, self._vectors)
        self._arc_start = line.arc_length
        self._arc_span = np.diff(line.arc_length, append=line.length)
        following = np.roll(np.arange(len(line.points)), -1)
        self._tangents = line.tangent, line.tangent[following]  # at each segment's two ends
        self._widths = track.width, track.width[following]
        spacing = line.length / len(line.points)
        steps = min(math.ceil((reach + MARGIN) / spacing), len(line.points) // 2)
        # The samples with the lap's ends repeated beyond them: sample i is at i + steps, and a
        # window of i - steps .. i + steps needs no wrapping round.
        padded = np.concatenate([line.points[-steps:], line.points, line.points[:steps]])
        self._x, self._y = padded[:, 0].copy(), padded[:, 1].copy()
        self._steps = steps
        self._window = np.arange(2 * steps + 1)

    def locate(self, points: ArrayLike, near: IntArray | None = None) -> RoadPosition:
        """Project points onto the centerline.

        Args:
            points: Shape ``(N, 2)``.
            near: Each point's segment from the last update, to search only around it; ``None``
                searches the whole centerline (for cars that just started somewhere).
        """
        xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if near is not None:
            return self._closest(xy, near[:, None] + self._window)
        everywhere = np.arange(len(self._starts)) + self._steps
        parts = [
            self._closest(chunk, np.broadcast_to(everywhere, (len(chunk), len(everywhere))))
            for chunk in np.split(xy, range(_CHUNK, len(xy), _CHUNK))
        ]
        return RoadPosition(*(np.concatenate(column) for column in zip(*parts, strict=True)))

    def _closest(self, points: FloatArray, candidates: IntArray) -> RoadPosition:
        """The closest point on the centerline to each point, near its candidate samples
        (indices into the padded samples, shape ``(N, K)``).

        Finding the nearest sample first and then measuring only to the two segments that meet
        there is much cheaper than measuring to every candidate segment. With samples 0.5 m
        apart along a smooth curve, it lands on the same segment.

        The road's heading and width are blended between the segment's two ends, so they change
        smoothly along the road. Taken per segment, they would step at every sample, and a point
        almost equally far from two segments (which rounding can tip either way) would get one
        step or the other.
        """
        rows = np.arange(len(points))
        dx = points[:, :1] - self._x[candidates]
        dy = points[:, 1:] - self._y[candidates]
        nearest = candidates[rows, np.argmin(dx * dx + dy * dy, axis=1)] - self._steps
        pair = np.column_stack([nearest - 1, nearest]) % len(self._starts)  # (N, 2) segments
        starts, vectors = self._starts[pair], self._vectors[pair]
        relative = points[:, None, :] - starts
        along = np.einsum("nki,nki->nk", relative, vectors) / self._squared_lengths[pair]
        fraction = np.clip(along, 0.0, 1.0)
        gaps = relative - fraction[..., None] * vectors
        best = np.argmin(np.einsum("nki,nki->nk", gaps, gaps), axis=1)
        segment = pair[rows, best]
        gap = gaps[rows, best]
        along_segment = fraction[rows, best]
        arc_length = self._arc_start[segment] + along_segment * self._arc_span[segment]
        side = np.where(cross(self._vectors[segment], gap) < 0, -1.0, 1.0)
        tangent = _blend(self._tangents, segment, along_segment[:, None])
        return RoadPosition(
            segment=segment,
            arc_length=np.where(arc_length >= self.length, arc_length - self.length, arc_length),
            offset=side * np.hypot(gap[:, 0], gap[:, 1]),
            heading=np.arctan2(tangent[:, 1], tangent[:, 0]),
            width=_blend(self._widths, segment, along_segment),
        )


def _blend(
    ends: tuple[FloatArray, FloatArray], segment: IntArray, fraction: FloatArray
) -> FloatArray:
    """A quantity known at both ends of each segment, ``fraction`` of the way along it."""
    start, end = ends
    result: FloatArray = start[segment] + (end[segment] - start[segment]) * fraction
    return result
