"""Drawing the world with pygame: the ground grid and the track.

Shared by the track editor and, later, the race window. Everything goes through a `Camera`, and
only the stretches of road on screen are filled, so a 3.5 km track draws in a few milliseconds
at any zoom (pygame's polygon fill costs time for every pixel row a polygon spans, on screen or
not).
"""

import math

import numpy as np
import pygame

from mlracecar.core.geometry import FloatArray
from mlracecar.core.track.model import Track
from mlracecar.render.camera import Camera

type Color = tuple[int, int, int]

BACKGROUND: Color = (28, 32, 36)
GRID_MINOR: Color = (35, 40, 45)
GRID_MAJOR: Color = (47, 53, 60)
ROAD: Color = (72, 77, 84)
EDGE: Color = (226, 229, 233)
START_LINE: Color = (255, 255, 255)
DIRECTION: Color = (250, 204, 21)

MAJOR_EVERY = 5
"""Every fifth grid line is drawn brighter."""

MAX_SAMPLE_GAP = 2.0
"""Pixels between the road samples that get filled; closer samples are skipped when zoomed out."""


def draw_grid(surface: pygame.Surface, camera: Camera, step: float) -> None:
    """Draw grid lines ``step`` metres apart over the visible area."""
    low, high = camera.visible_area()
    width, height = camera.size
    for axis in (0, 1):
        first, last = math.ceil(low[axis] / step), math.floor(high[axis] / step)
        for index in range(first, last + 1):
            color = GRID_MAJOR if index % MAJOR_EVERY == 0 else GRID_MINOR
            if axis == 0:
                x = float(camera.to_screen((index * step, 0.0))[0])
                pygame.draw.line(surface, color, (x, 0), (x, height))
            else:
                y = float(camera.to_screen((0.0, index * step))[1])
                pygame.draw.line(surface, color, (0, y), (width, y))


def draw_track(surface: pygame.Surface, track: Track, camera: Camera) -> None:
    """Draw the road with its edges, the start/finish line, and an arrow for the direction."""
    gap = track.length / len(track.left) * camera.scale
    every = max(1, math.floor(MAX_SAMPLE_GAP / gap))
    left = camera.to_screen(track.left[::every])
    right = camera.to_screen(track.right[::every])
    for strip in visible_strips(left, right, camera.size):
        pygame.draw.polygon(surface, ROAD, strip)

    pygame.draw.aalines(surface, EDGE, True, camera.to_screen(track.left))
    pygame.draw.aalines(surface, EDGE, True, camera.to_screen(track.right))
    start = camera.to_screen([track.checkpoints.left[0], track.checkpoints.right[0]])
    pygame.draw.line(surface, START_LINE, start[0], start[1], width=3)
    _draw_direction(surface, track, camera)


def visible_strips(left: FloatArray, right: FloatArray, size: tuple[int, int]) -> list[FloatArray]:
    """Polygons covering the parts of a closed road that are on screen.

    The road is cut into pieces between neighbouring samples; each run of on-screen pieces
    becomes one polygon. A road that's entirely on screen is one ring (out along one edge and
    back along the other); pygame fills it even-odd, so the infield stays empty.

    Args:
        left: Left edge in pixels, one point per sample, shape ``(n, 2)``.
        right: Right edge in pixels, shape ``(n, 2)``.
        size: Window width and height in pixels.
    """
    count = len(left)
    following = np.roll(np.arange(count), -1)
    corners = np.stack([left, left[following], right[following], right])
    low, high = corners.min(axis=0), corners.max(axis=0)
    on_screen = np.all((high >= 0) & (low <= size), axis=1)
    if not on_screen.any():
        return []
    if on_screen.all():
        return [np.concatenate([left, left[:1], right[:1], right[::-1]])]

    # Start the walk at an off-screen piece so no run wraps around the end of the arrays.
    order = np.roll(np.arange(count), -int(np.argmin(on_screen)))
    edges = np.diff(np.concatenate([[0], on_screen[order].astype(np.int8), [0]]))
    strips = []
    for begin, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        pieces = order[begin:end]
        samples = np.append(pieces, following[pieces[-1]])
        strips.append(np.concatenate([left[samples], right[samples[::-1]]]))
    return strips


def _draw_direction(surface: pygame.Surface, track: Track, camera: Camera) -> None:
    """A chevron one road width past the start line, pointing the way the track is driven."""
    width = float(track.width[0])
    index = min(len(track.left) - 1, round(width / (track.length / len(track.left))))
    tip_length = max(8.0, 0.4 * width * camera.scale)
    center = camera.to_screen(track.centerline.points[index])
    ahead = track.centerline.tangent[index] * (1.0, -1.0)  # screen y points down
    side = np.array([-ahead[1], ahead[0]])
    tip = center + ahead * tip_length / 2
    back = center - ahead * tip_length / 2
    pygame.draw.polygon(
        surface, DIRECTION, [tip, back + side * tip_length / 2, back - side * tip_length / 2]
    )
