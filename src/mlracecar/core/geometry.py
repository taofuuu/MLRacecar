"""Vectorized 2D geometry: the math the rest of the simulator is built on.

Every function works on batches. Points and vectors are arrays whose last axis holds
``(x, y)``, so one call handles every car, sensor ray, or track segment at once instead of
looping in Python. Inputs broadcast like ordinary NumPy arithmetic.

Conventions (architecture.md section 4.1): metres and radians, x to the right, y up, angles
counter-clockwise from +x. "Left of a direction" means counter-clockwise from it.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.intp]

# Two directions count as parallel when |cross(a, b)| <= PARALLEL_TOLERANCE * |a| * |b|,
# i.e. when the angle between them is below about 1e-12 radians.
PARALLEL_TOLERANCE = 1e-12

# Hits this far past a segment's end (as a fraction of its length) still count. Without it,
# rounding lets a ray aimed exactly at the shared corner of two connected segments miss both,
# so a sensor could see "through" a track boundary.
ENDPOINT_TOLERANCE = 1e-9

# Segments per block when searching for crossings. Each block is checked against only the
# segments near it (see `_crossing_pairs`); 64 measured fastest on 0.6-3 km tracks.
_CROSSING_BLOCK = 64


def _as_vectors(values: ArrayLike, name: str) -> FloatArray:
    """Convert to a float64 array whose last axis has length 2."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2), got {array.shape}")
    return array


def _as_polyline(vertices: ArrayLike) -> FloatArray:
    """Convert to a float64 array of shape (V, 2) with at least two vertices."""
    array = _as_vectors(vertices, "vertices")
    if array.ndim != 2 or len(array) < 2:
        raise ValueError(f"vertices must have shape (V, 2) with V >= 2, got {array.shape}")
    return array


def _polyline_segments(vertices: FloatArray, closed: bool) -> tuple[FloatArray, FloatArray]:
    """Start and end points of each segment; closed polylines include last -> first."""
    if closed:
        return vertices, np.roll(vertices, -1, axis=0)
    return vertices[:-1], vertices[1:]


def cross(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """2D cross product ``a.x * b.y - a.y * b.x`` (the z part of the 3D cross product).

    Positive when ``b`` points counter-clockwise (to the left) of ``a``, negative when
    clockwise, zero when they are parallel.

    Args:
        a: Vectors of shape ``(..., 2)``.
        b: Vectors of shape ``(..., 2)``, broadcastable against ``a``.

    Returns:
        Array of shape ``(...)``.
    """
    a = _as_vectors(a, "a")
    b = _as_vectors(b, "b")
    result: FloatArray = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return result


def norm(vectors: ArrayLike) -> FloatArray:
    """Length of each vector.

    Args:
        vectors: Vectors of shape ``(..., 2)``.

    Returns:
        Array of shape ``(...)``.
    """
    v = _as_vectors(vectors, "vectors")
    result: FloatArray = np.hypot(v[..., 0], v[..., 1])
    return result


def wrap_angle(angles: ArrayLike) -> FloatArray:
    """Map angles to the equivalent angle in ``[-pi, pi)``.

    Args:
        angles: Angles in radians, any shape.

    Returns:
        Array of the same shape, pointing the same way as the input. Angles already in range
        come back exactly as they were.
    """
    a = np.asarray(angles, dtype=np.float64)
    wrapped = np.mod(a + np.pi, 2 * np.pi) - np.pi
    # Rounding can land exactly on +pi (e.g. for inputs just below -pi); fold it back.
    wrapped = np.where(wrapped >= np.pi, wrapped - 2 * np.pi, wrapped)
    # Adding and subtracting pi rounds too (0.7 becomes 0.7000000000000002), which would
    # make a car driving straight ahead drift off its heading, step by step.
    result: FloatArray = np.where((a >= -np.pi) & (a < np.pi), a, wrapped)
    return result


def unit_vector(angles: ArrayLike) -> FloatArray:
    """Unit vectors pointing in the direction of each angle.

    Args:
        angles: Angles in radians, any shape.

    Returns:
        Array of shape ``(*angles.shape, 2)``.
    """
    a = np.asarray(angles, dtype=np.float64)
    return np.stack([np.cos(a), np.sin(a)], axis=-1)


def rotate(vectors: ArrayLike, angles: ArrayLike) -> FloatArray:
    """Rotate vectors counter-clockwise.

    Args:
        vectors: Vectors of shape ``(..., 2)``.
        angles: Rotation in radians, broadcastable against ``vectors[..., 0]``.

    Returns:
        Rotated vectors, same shape as the broadcast inputs.
    """
    v = _as_vectors(vectors, "vectors")
    a = np.asarray(angles, dtype=np.float64)
    c, s = np.cos(a), np.sin(a)
    x, y = v[..., 0], v[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def segments_intersect(
    a_start: ArrayLike, a_end: ArrayLike, b_start: ArrayLike, b_end: ArrayLike
) -> BoolArray:
    """Whether segment ``a`` crosses or touches segment ``b``.

    Parallel segments (including overlapping collinear ones and zero-length segments) count
    as not intersecting. The result is symmetric: swapping ``a`` and ``b`` gives the same
    answer.

    Args:
        a_start: Start points of the first segments, shape ``(..., 2)``.
        a_end: End points of the first segments, shape ``(..., 2)``.
        b_start: Start points of the second segments, shape ``(..., 2)``.
        b_end: End points of the second segments, shape ``(..., 2)``.

    Returns:
        Boolean array with the broadcast shape of the inputs (without the last axis).
    """
    p = _as_vectors(a_start, "a_start")
    r = _as_vectors(a_end, "a_end") - p
    q = _as_vectors(b_start, "b_start")
    s = _as_vectors(b_end, "b_end") - q
    qp = q - p
    denominator = cross(r, s)
    parallel = np.abs(denominator) <= PARALLEL_TOLERANCE * norm(r) * norm(s)
    # Near-parallel or degenerate inputs divide by (almost) zero; the resulting inf/nan values
    # fail the range checks below, which is the right answer, so the warnings are silenced.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = cross(qp, s) / denominator  # position along a, 0..1 inside the segment
        u = cross(qp, r) / denominator  # position along b, 0..1 inside the segment
    result: BoolArray = ~parallel & _within_segment(t) & _within_segment(u)
    return result


def _within_segment(fraction: FloatArray) -> BoolArray:
    """Whether a position along a segment (0 = start, 1 = end) lies on it, up to tolerance."""
    result: BoolArray = (fraction >= -ENDPOINT_TOLERANCE) & (fraction <= 1 + ENDPOINT_TOLERANCE)
    return result


def ray_segment_distances(
    origins: ArrayLike,
    directions: ArrayLike,
    seg_start: ArrayLike,
    seg_end: ArrayLike,
) -> FloatArray:
    """Distance along each ray to the point where it hits each segment.

    Rays start at ``origins`` and travel along ``directions`` (any non-zero length; the
    distance is measured in metres, not in multiples of the direction). Inputs broadcast, so
    ``origins[:, None]`` against ``seg_start[None, :]`` gives every ray against every segment.

    Args:
        origins: Ray start points, shape ``(..., 2)``.
        directions: Ray directions, shape ``(..., 2)``.
        seg_start: Segment start points, shape ``(..., 2)``.
        seg_end: Segment end points, shape ``(..., 2)``.

    Returns:
        Distances with the broadcast shape of the inputs (without the last axis); ``inf``
        where the ray misses the segment or runs parallel to it.
    """
    p = _as_vectors(origins, "origins")
    r = _as_vectors(directions, "directions")
    q = _as_vectors(seg_start, "seg_start")
    s = _as_vectors(seg_end, "seg_end") - q
    qp = q - p
    denominator = cross(r, s)
    ray_length = norm(r)
    parallel = np.abs(denominator) <= PARALLEL_TOLERANCE * ray_length * norm(s)
    # Near-parallel or degenerate inputs divide by (almost) zero; the resulting inf/nan values
    # fail the range checks below, which is the right answer, so the warnings are silenced.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = cross(qp, s) / denominator  # distance along the ray, in direction lengths
        u = cross(qp, r) / denominator  # position along the segment, 0..1 inside it
        hit = ~parallel & (t >= 0) & _within_segment(u)
        result: FloatArray = np.where(hit, t * ray_length, np.inf)
    return result


def cast_rays(
    origins: ArrayLike,
    directions: ArrayLike,
    seg_start: ArrayLike,
    seg_end: ArrayLike,
    max_distance: float = np.inf,
) -> FloatArray:
    """Distance from each ray's origin to the nearest segment it hits.

    This is the core of the car's distance sensors: every ray is tested against every
    segment in one vectorized step, and the closest hit wins.

    Args:
        origins: Ray start points, shape ``(M, 2)``.
        directions: Ray directions, shape ``(M, 2)``.
        seg_start: Segment start points, shape ``(S, 2)``.
        seg_end: Segment end points, shape ``(S, 2)``.
        max_distance: Sensor range; rays that hit nothing closer report this value.

    Returns:
        Array of shape ``(M,)``.
    """
    o = _as_vectors(origins, "origins")
    d = _as_vectors(directions, "directions")
    a = _as_vectors(seg_start, "seg_start")
    b = _as_vectors(seg_end, "seg_end")
    distances = ray_segment_distances(o[:, None, :], d[:, None, :], a[None, :, :], b[None, :, :])
    nearest = distances.min(axis=1, initial=np.inf)
    result: FloatArray = np.minimum(nearest, max_distance)
    return result


class PolylineProjection(NamedTuple):
    """Where points land when projected onto a polyline. All fields have one entry per point."""

    point: FloatArray
    """Closest point on the polyline, shape ``(N, 2)``."""
    segment: IntArray
    """Index of the segment that holds the closest point, shape ``(N,)``."""
    arc_length: FloatArray
    """Distance along the polyline from its first vertex to the closest point, shape ``(N,)``."""
    offset: FloatArray
    """Signed distance from the polyline: positive to the left of the direction of travel,
    negative to the right. ``abs(offset)`` is the distance to the polyline. Points exactly in
    line with their closest segment (beyond its end) count as left. Shape ``(N,)``."""


def project_onto_polyline(
    points: ArrayLike, vertices: ArrayLike, *, closed: bool
) -> PolylineProjection:
    """Find the closest point on a polyline for each query point.

    Used for race progress: projecting a car onto the track centerline gives how far around
    the lap it is (``arc_length``) and how far it is from the middle of the road (``offset``).
    Every point is compared with every segment at once; when two segments are equally close,
    the lower segment index wins.

    Args:
        points: Query points, shape ``(N, 2)``.
        vertices: Polyline vertices in order of travel, shape ``(V, 2)`` with ``V >= 2``.
        closed: Whether the polyline is a loop (an extra segment joins the last vertex back
            to the first).

    Returns:
        The projection of every point.
    """
    pts = _as_vectors(points, "points")
    if pts.ndim != 2:
        raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
    starts, ends = _polyline_segments(_as_polyline(vertices), closed)
    segments = ends - starts
    squared_lengths = np.einsum("si,si->s", segments, segments)

    relative = pts[:, None, :] - starts[None, :, :]  # (N, S, 2)
    along = np.einsum("nsi,si->ns", relative, segments)
    # Fraction along each segment of the closest point; zero-length segments use their start.
    fraction = np.divide(
        along, squared_lengths, out=np.zeros_like(along), where=squared_lengths > 0
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    candidates = starts[None, :, :] + fraction[..., None] * segments[None, :, :]
    gaps = pts[:, None, :] - candidates
    squared_distances = np.einsum("nsi,nsi->ns", gaps, gaps)

    best = np.argmin(squared_distances, axis=1)
    rows = np.arange(len(pts))
    closest = candidates[rows, best]
    lengths = np.sqrt(squared_lengths)
    distance_to_segment_start = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
    arc_length = distance_to_segment_start[best] + fraction[rows, best] * lengths[best]
    # The side is ambiguous when the point is exactly in line with the segment (e.g. beyond
    # its end) or the segment has zero length; report those on the left so that
    # abs(offset) always equals the distance.
    side = np.where(cross(segments[best], pts - closest) < 0, -1.0, 1.0)
    offset = side * np.sqrt(squared_distances[rows, best])
    return PolylineProjection(closest, best, arc_length, offset)


def self_intersections(vertices: ArrayLike, *, closed: bool) -> IntArray:
    """Find pairs of polyline segments that cross or touch each other.

    Neighbouring segments always share a vertex, so they are never reported. Used to reject
    tracks that cross themselves.

    Args:
        vertices: Polyline vertices in order, shape ``(V, 2)`` with ``V >= 2``.
        closed: Whether the polyline is a loop.

    Returns:
        Array of shape ``(K, 2)`` with segment index pairs ``(i, j)``, ``i < j``, sorted.
        Segment ``i`` runs from ``vertices[i]`` to the next vertex.
    """
    starts, ends = _polyline_segments(_as_polyline(vertices), closed)
    count = len(starts)

    def keep(i: IntArray, j: IntArray) -> BoolArray:
        neighbours = (j == i + 1) | (closed & (i == 0) & (j == count - 1))
        result: BoolArray = (j > i) & ~neighbours
        return result

    return _crossing_pairs(starts, ends, starts, ends, keep)


def polyline_crossings(first: ArrayLike, second: ArrayLike, *, closed: bool) -> IntArray:
    """Find pairs of segments, one from each polyline, that cross or touch.

    Args:
        first: Vertices of the first polyline, shape ``(V, 2)`` with ``V >= 2``.
        second: Vertices of the second polyline, shape ``(W, 2)`` with ``W >= 2``.
        closed: Whether both polylines are loops.

    Returns:
        Array of shape ``(K, 2)`` with pairs ``(i, j)``: segment ``i`` of ``first`` touches
        segment ``j`` of ``second``. Sorted.
    """
    a_start, a_end = _polyline_segments(_as_polyline(first), closed)
    b_start, b_end = _polyline_segments(_as_polyline(second), closed)
    return _crossing_pairs(a_start, a_end, b_start, b_end, lambda i, j: np.ones((), np.bool_))


def _crossing_pairs(
    a_start: FloatArray,
    a_end: FloatArray,
    b_start: FloatArray,
    b_end: FloatArray,
    keep: Callable[[IntArray, IntArray], BoolArray],
) -> IntArray:
    """Index pairs ``(i, j)`` where segment ``a[i]`` touches segment ``b[j]`` and ``keep`` agrees.

    Works through ``a`` in blocks of consecutive segments to bound memory. For each block, a
    broad phase first keeps only the ``b`` segments whose bounding boxes overlap the block's
    bounding box: on a track, a stretch of road can only touch road nearby, so this skips most
    pairs. The survivors get the exact, vectorized test.

    Boxes are padded by `ENDPOINT_TOLERANCE` times the segment length, the same margin the
    exact test accepts past a segment's ends, so the broad phase never drops a pair the exact
    test would report.
    """
    b_pad = (ENDPOINT_TOLERANCE * norm(b_end - b_start))[:, None]
    b_low = np.minimum(b_start, b_end) - b_pad
    b_high = np.maximum(b_start, b_end) + b_pad
    a_pad = ENDPOINT_TOLERANCE * norm(a_end - a_start)
    found: list[IntArray] = [np.empty((0, 2), dtype=np.intp)]
    for first in range(0, len(a_start), _CROSSING_BLOCK):
        i = np.arange(first, min(first + _CROSSING_BLOCK, len(a_start)))
        pad = a_pad[i].max()
        block_low = np.minimum(a_start[i], a_end[i]).min(axis=0) - pad
        block_high = np.maximum(a_start[i], a_end[i]).max(axis=0) + pad
        j = np.flatnonzero(np.all((b_low <= block_high) & (b_high >= block_low), axis=1))
        if len(j) == 0:
            continue
        touching = segments_intersect(
            a_start[i][:, None], a_end[i][:, None], b_start[j][None, :], b_end[j][None, :]
        )
        rows, columns = np.nonzero(touching & keep(i[:, None], j[None, :]))
        found.append(np.column_stack([i[rows], j[columns]]))
    return np.concatenate(found).astype(np.intp)
