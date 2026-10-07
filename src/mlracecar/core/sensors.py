"""Distance sensors: how a car sees the road (architecture section 4.6).

Rays fan out from each car's centre across a field of view, like a lidar, and stop where they
first meet an edge of the road. Each reading is the distance to that edge, up to the sensor's
range.

The road's edges are polylines with a point every half metre, thousands of them per lap. Testing
every ray against every piece of edge would be far too slow, so the work narrows down in steps:

1. **Only the edges within reach.** A ray from a car on the road meets one of its road's edges
   before it gets more than its range along the lap. So each car's rays are tested only against
   the edges from ``range + MARGIN`` metres behind its place on the lap to as far ahead. That
   also means a car out on the grass doesn't see other parts of the track that pass close by,
   such as the far side of a hairpin.
2. **Only the blocks a ray passes close to.** The edges are cut into blocks of `BLOCK` pieces,
   each inside a circle worked out once per track. A ray is tested against a block's pieces only
   if it passes within that circle.
3. **Only the pieces a ray crosses.** A ray can only meet a piece whose two ends lie on opposite
   sides of the ray's line. A sign test per point finds those pieces; only they get the exact
   intersection.

Steps 1 and 2 only skip work: the readings are the same as testing every piece of edge within
reach.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from mlracecar.core.geometry import FloatArray, IntArray, unit_vector
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track

MARGIN = 10.0
"""Metres of edge searched beyond the sensor's range, both ahead and behind."""

BLOCK = 16
"""Pieces of edge per block (8 m with the usual half-metre spacing)."""

_CHUNK_SIZE = 500_000
"""Ray-and-block pairs worked out at once, to keep memory use small with thousands of cars."""


@dataclass(frozen=True)
class RaySettings:
    """The sensor's settings. `mlracecar.config.models.SensorConfig` builds them from settings
    files; the defaults here are the same."""

    count: int = 15
    """How many rays."""
    field_of_view: float = math.pi
    """The angle the rays fan across, in radians, centred on the car's heading."""
    max_range: float = 100.0
    """How far the rays reach, in metres."""


class RayReadings(NamedTuple):
    """What the rays saw: one row per car, one column per ray (in the order of ``angles``)."""

    distance: FloatArray
    """Distance to the road's edge in metres, or the range where it's further; shape ``(N, R)``."""
    normalized: FloatArray
    """The distances as a fraction of the range, in ``[0, 1]``; shape ``(N, R)``."""
    end: FloatArray
    """Where each ray stops, shape ``(N, R, 2)``; for drawing them."""


class _Hits(NamedTuple):
    """Where rays cross pieces of edge: one entry per crossing."""

    ray: IntArray
    """Which ray, as ``car * R + ray``."""
    along: FloatArray
    """How far along the ray's line the crossing is, in metres; negative behind the car."""


class RaySensor:
    """Distance rays for every car on a track.

    Args:
        track: The track whose road edges the rays meet.
        settings: How many rays, how wide they fan, and how far they reach.

    Raises:
        ValueError: If there are no rays, the field of view isn't above 0 and at most a full
            circle, or the range isn't positive.
    """

    def __init__(self, track: Track, settings: RaySettings | None = None) -> None:
        settings = settings or RaySettings()
        if settings.count < 1:
            raise ValueError(f"need at least 1 ray, got {settings.count}")
        if not 0 < settings.field_of_view <= 2 * math.pi:
            raise ValueError(
                f"the field of view must be above 0 and at most 2 pi, got {settings.field_of_view}"
            )
        if not settings.max_range > 0:
            raise ValueError(f"the range must be positive, got {settings.max_range}")
        self.settings = settings
        self.angles = _ray_angles(settings.count, settings.field_of_view)
        """Each ray's direction relative to the car's heading, in radians, shape ``(R,)``;
        counter-clockwise, so the first ray is the rightmost."""

        points = len(track.left)
        reach = min(math.ceil((settings.max_range + MARGIN) * points / track.length), points // 2)
        blocks = -(-points // BLOCK)
        # Each block's points, both edges side by side: (blocks, BLOCK + 1, 2 edges, 2). A block
        # ends where the next one starts; the last one runs on round to the start of the lap.
        index = (np.arange(blocks)[:, None] * BLOCK + np.arange(BLOCK + 1)) % points
        self._points = np.stack([track.left, track.right], axis=1)[index]
        lowest, highest = self._points.min(axis=1), self._points.max(axis=1)
        self._centers = (lowest + highest) / 2  # (blocks, 2 edges, 2)
        offsets = self._points - self._centers[:, None]
        self._radii = np.sqrt(np.einsum("bpei,bpei->bpe", offsets, offsets).max(axis=1))
        # A car on piece k needs pieces k - reach .. k + reach: these blocks, counted from the
        # one holding piece k - reach.
        self._reach = reach
        self._window = np.arange((2 * reach) // BLOCK + 2)

    def sense(self, snapshot: Snapshot) -> RayReadings:
        """Cast every car's rays.

        Each car's rays start at its centre. The car's place along the lap (``race.segment``)
        picks the stretch of edge they are tested against.
        """
        cars = snapshot.cars
        count, rays = len(cars), len(self.angles)
        directions = unit_vector(cars.yaw[:, None] + self.angles)  # (N, R, 2)
        first = (snapshot.race.segment - self._reach) // BLOCK
        blocks = (first[:, None] + self._window) % len(self._points)  # (N, W)
        size = max(1, _CHUNK_SIZE // (rays * self._window.size * 2))
        nearest = np.full(count * rays, self.settings.max_range)
        for start in range(0, count, size):
            cars_here = slice(start, start + size)
            hits = self._hits(cars.position[cars_here], directions[cars_here], blocks[cars_here])
            ahead = hits.along >= 0  # the ray's line also runs behind the car
            np.minimum.at(nearest, start * rays + hits.ray[ahead], hits.along[ahead])
        distance = nearest.reshape(count, rays)
        return RayReadings(
            distance=distance,
            normalized=distance / self.settings.max_range,
            end=cars.position[:, None, :] + directions * distance[..., None],
        )

    def _hits(self, origins: FloatArray, directions: FloatArray, blocks: IntArray) -> _Hits:
        """Every crossing of a ray's line with a piece of edge in a block it passes close to.

        Args:
            origins: Each car's position, shape ``(n, 2)``.
            directions: Each ray's unit direction, shape ``(n, R, 2)``.
            blocks: The blocks within each car's reach, shape ``(n, W)``.
        """
        count, rays = directions.shape[:2]
        # Step 2: which blocks' circles each ray passes through, measured across the ray's line
        # and along it, for every ray and block at once as matrix products: (n, R, W * 2 edges).
        centers = (self._centers[blocks] - origins[:, None, None, :]).reshape(count, -1, 2)
        across = directions @ _turned(centers)
        along = directions @ centers.transpose(0, 2, 1)
        radii = self._radii[blocks].reshape(count, 1, -1)
        close = (np.abs(across) <= radii) & (along >= -radii)
        close &= along <= self.settings.max_range + radii
        car, ray, slot = np.nonzero(close)
        direction = directions[car, ray]  # (C, 2)
        points = self._points[blocks[car, slot // 2], :, slot % 2] - origins[car][:, None, :]
        # Step 3: the side of the ray's line each point is on, cross(direction, point), and the
        # pieces whose ends are on opposite sides.
        side = direction[:, None, 0] * points[..., 1] - direction[:, None, 1] * points[..., 0]
        left = side >= 0
        pair, piece = np.nonzero(left[:, 1:] != left[:, :-1])
        # Where those pieces meet the ray's line. Their ends' sides have opposite signs, so the
        # division is safe.
        before, after = side[pair, piece], side[pair, piece + 1]
        start, end = points[pair, piece], points[pair, piece + 1]
        meeting = start + (before / (before - after))[:, None] * (end - start)
        return _Hits(
            ray=car[pair] * rays + ray[pair],
            along=np.einsum("pi,pi->p", meeting, direction[pair]),
        )


def _turned(vectors: FloatArray) -> FloatArray:
    """``(y, -x)`` for each vector of shape ``(n, k, 2)``, laid out as ``(n, 2, k)`` so that a
    matrix product with directions gives ``cross(direction, vector)``."""
    return np.stack([vectors[..., 1], -vectors[..., 0]], axis=1)


def _ray_angles(count: int, field_of_view: float) -> FloatArray:
    """Angles spread evenly across the field of view, centred on the car's heading. A full
    circle leaves out the last angle, which would repeat the first."""
    if count == 1:
        return np.zeros(1)
    full_circle = math.isclose(field_of_view, 2 * math.pi)
    return np.linspace(
        -field_of_view / 2, field_of_view / 2, count, endpoint=not full_circle, dtype=np.float64
    )
