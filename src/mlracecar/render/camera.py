"""Which part of the world a window shows, and how big: converting metres to pixels and back.

The world has y pointing up (architecture.md section 4.1) and the screen has y pointing down, so
the camera flips y. A `Camera` is immutable like the editor's drafts: panning or zooming returns
a new camera. There's no pygame here, so the editor's controller can use it headless.
"""

import math
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray

MIN_SCALE = 0.05
"""Most zoomed out, in pixels per metre: a 1280-pixel window then spans 25.6 km."""

MAX_SCALE = 50.0
"""Most zoomed in, in pixels per metre: a 1280-pixel window then spans 25.6 m."""

DEFAULT_SCALE = 2.0
"""Starting zoom, in pixels per metre: a 1280-pixel window spans 640 m."""

NORTH = math.pi / 2
"""The world's +y direction: up the window for a camera that doesn't turn."""


@dataclass(frozen=True)
class Camera:
    """A view of the world: the point at the window's centre, the zoom, and the window size."""

    center: tuple[float, float] = (0.0, 0.0)
    """World position shown at the centre of the window, in metres."""
    scale: float = DEFAULT_SCALE
    """Zoom, in pixels per metre, kept between `MIN_SCALE` and `MAX_SCALE`."""
    size: tuple[int, int] = (1280, 800)
    """Window width and height, in pixels."""
    up: float = NORTH
    """World direction shown pointing up the window, in radians (counter-clockwise from +x).
    `NORTH`, the default, doesn't turn the view; the chase camera turns it with the car."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", float(np.clip(self.scale, MIN_SCALE, MAX_SCALE)))

    # ------------------------------------------------------------------ #
    # Converting positions
    # ------------------------------------------------------------------ #

    def to_screen(self, points: ArrayLike) -> FloatArray:
        """World positions (metres, shape ``(..., 2)``) to window pixels."""
        world = np.asarray(points, dtype=np.float64)
        screen = _turned(world - self.center, NORTH - self.up) * self.scale * (1.0, -1.0)
        return screen + np.asarray(self.size) / 2

    def to_world(self, pixels: ArrayLike) -> FloatArray:
        """Window pixels (shape ``(..., 2)``) to world positions in metres."""
        screen = np.asarray(pixels, dtype=np.float64) - np.asarray(self.size) / 2
        return _turned(screen * (1.0, -1.0) / self.scale, self.up - NORTH) + self.center

    def visible_area(self) -> tuple[FloatArray, FloatArray]:
        """The lowest and highest world corners the window shows, as ``(x, y)`` arrays."""
        width, height = self.size
        corners = self.to_world([(0, 0), (width, 0), (0, height), (width, height)])
        return corners.min(axis=0), corners.max(axis=0)

    # ------------------------------------------------------------------ #
    # Moving the camera
    # ------------------------------------------------------------------ #

    def pan(self, dx: float, dy: float) -> Self:
        """Drag the view by ``(dx, dy)`` pixels: the world moves with the mouse."""
        moved = _turned(np.array([dx / self.scale, -dy / self.scale]), self.up - NORTH)
        x, y = self.center
        return replace(self, center=(x - float(moved[0]), y - float(moved[1])))

    def zoom_at(self, pixel: ArrayLike, factor: float) -> Self:
        """Zoom by ``factor`` (above 1 zooms in), keeping the world point under ``pixel`` still."""
        anchor = self.to_world(pixel)
        zoomed = replace(self, scale=self.scale * factor)
        drift = anchor - zoomed.to_world(pixel)
        x, y = zoomed.center
        return replace(zoomed, center=(x + float(drift[0]), y + float(drift[1])))

    def fit(self, points: ArrayLike, margin: float = 40.0) -> Self:
        """Centre on ``points`` (shape ``(N, 2)``) and zoom so they all fit, ``margin`` px in."""
        world = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(world) == 0:
            return replace(self, center=(0.0, 0.0), scale=DEFAULT_SCALE)
        low, high = world.min(axis=0), world.max(axis=0)
        room = np.maximum(np.asarray(self.size) - 2 * margin, 1.0)
        extent = np.maximum(high - low, 1e-9)
        center = (low + high) / 2
        return replace(
            self, center=(float(center[0]), float(center[1])), scale=float(np.min(room / extent))
        )

    def resized(self, size: tuple[int, int]) -> Self:
        """The same view in a window of a new size."""
        return replace(self, size=size)

    # ------------------------------------------------------------------ #
    # Grid
    # ------------------------------------------------------------------ #

    def grid_step(self, min_pixels: float = 16.0) -> float:
        """The finest grid spacing (1, 2, or 5 times a power of ten, in metres) whose lines are
        at least ``min_pixels`` apart at this zoom."""
        smallest = min_pixels / self.scale
        power = math.floor(math.log10(smallest))
        steps = (1, power), (2, power), (5, power), (1, power + 1)
        return next(step for m, p in steps if (step := m * 10.0**p) >= smallest)


def _turned(points: FloatArray, angle: float) -> FloatArray:
    """``points`` (shape ``(..., 2)``) turned ``angle`` radians counter-clockwise about 0."""
    if angle == 0.0:
        return points
    cos, sin = math.cos(angle), math.sin(angle)
    x, y = points[..., 0], points[..., 1]
    return np.stack([x * cos - y * sin, x * sin + y * cos], axis=-1)
