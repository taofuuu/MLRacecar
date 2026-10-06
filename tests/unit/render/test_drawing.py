"""Tests for mlracecar.render.drawing: the grid and the track, drawn onto images in memory."""

import numpy as np
import pygame
import pytest

from mlracecar.core.track.model import Track
from mlracecar.render.camera import Camera
from mlracecar.render.drawing import (
    BACKGROUND,
    DIRECTION,
    GRID_MAJOR,
    GRID_MINOR,
    ROAD,
    START_LINE,
    draw_grid,
    draw_track,
    visible_strips,
)

SIZE = (800, 600)


def oval_track(count: int = 16, width: float = 12.0) -> Track:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return Track.build(
        np.column_stack([120 * np.cos(angles), 60 * np.sin(angles)]), [width] * count
    )


def color_at(surface: pygame.Surface, pixel: object) -> tuple[int, int, int]:
    x, y = np.round(np.asarray(pixel, dtype=float)).astype(int)
    r, g, b, _ = surface.get_at((int(x), int(y)))
    return r, g, b


def blank() -> pygame.Surface:
    surface = pygame.Surface(SIZE)
    surface.fill(BACKGROUND)
    return surface


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


def test_grid_lines_fall_on_round_numbers_and_every_fifth_is_brighter() -> None:
    surface = blank()
    camera = Camera(scale=2.0, size=SIZE)  # 10 m steps are 20 px apart
    draw_grid(surface, camera, 10.0)
    assert color_at(surface, camera.to_screen((0.0, 3.0))) == GRID_MAJOR  # x = 0 m
    assert color_at(surface, camera.to_screen((10.0, 3.0))) == GRID_MINOR
    assert color_at(surface, camera.to_screen((50.0, 3.0))) == GRID_MAJOR
    assert color_at(surface, camera.to_screen((3.0, 20.0))) == GRID_MINOR  # y = 20 m
    assert color_at(surface, camera.to_screen((3.0, 3.0))) == BACKGROUND


# --------------------------------------------------------------------------- #
# Track
# --------------------------------------------------------------------------- #


def test_the_road_is_filled_and_the_infield_is_not() -> None:
    surface = blank()
    track = oval_track()
    camera = Camera(size=SIZE).fit(track.left)
    draw_track(surface, track, camera)
    assert color_at(surface, camera.to_screen((0.0, 60.0))) == ROAD  # on the top straight
    assert color_at(surface, camera.to_screen((0.0, 0.0))) == BACKGROUND  # the infield
    assert color_at(surface, camera.to_screen((0.0, 90.0))) == BACKGROUND  # outside


@pytest.mark.parametrize("zoom", [0.3, 4.0, 25.0])
def test_the_road_is_filled_at_any_zoom(zoom: float) -> None:
    surface = blank()
    track = oval_track()
    on_road = (-60.0, -52.0)  # inside the bottom straight, away from any edge line
    camera = Camera(center=on_road, scale=zoom, size=SIZE)
    draw_track(surface, track, camera)
    assert color_at(surface, (400, 300)) == ROAD


def is_bright(color: tuple[int, int, int]) -> bool:
    """Edge lines are anti-aliased, so their pixels are a blend; all of them are bright."""
    return min(color) > 150


def test_edges_start_line_and_direction_arrow_are_drawn() -> None:
    surface = blank()
    track = oval_track()
    camera = Camera(center=(120.0, 0.0), scale=10.0, size=SIZE)  # zoomed in on the start
    draw_track(surface, track, camera)
    assert is_bright(color_at(surface, camera.to_screen(track.left[20])))
    assert is_bright(color_at(surface, camera.to_screen(track.right[20])))
    start = (track.checkpoints.left[0] + track.checkpoints.right[0]) / 2
    assert color_at(surface, camera.to_screen(start)) == START_LINE
    # The arrow sits about one road width (12 m) past the start, the way the track goes.
    ahead = track.pose_at(12.0).position
    assert color_at(surface, camera.to_screen(ahead)) == DIRECTION


def test_reversing_the_track_turns_the_arrow_around() -> None:
    track = oval_track()
    points = track.spline.control_points
    reversed_track = Track.build(np.concatenate([points[:1], points[:0:-1]]), track.control_widths)
    surface = blank()
    camera = Camera(center=(120.0, 0.0), scale=10.0, size=SIZE)
    draw_track(surface, reversed_track, camera)
    assert color_at(surface, camera.to_screen(reversed_track.pose_at(12.0).position)) == DIRECTION
    assert color_at(surface, camera.to_screen(track.pose_at(12.0).position)) == ROAD


def test_a_track_off_screen_draws_nothing() -> None:
    surface = blank()
    draw_track(surface, oval_track(), Camera(center=(5000.0, 5000.0), size=SIZE))
    assert pygame.mask.from_threshold(surface, BACKGROUND, (1, 1, 1, 255)).count() == 800 * 600


# --------------------------------------------------------------------------- #
# Visible strips
# --------------------------------------------------------------------------- #


def ring(count: int, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.add(center, 110 * unit), np.add(center, 90 * unit)


def test_a_road_fully_on_screen_is_one_ring() -> None:
    left, right = ring(40, (400, 300))
    (strip,) = visible_strips(left, right, SIZE)
    assert len(strip) == 2 * 40 + 2  # out along one edge, across, and back along the other


def test_a_road_crossing_the_window_edge_is_cut_to_the_part_on_screen() -> None:
    left, right = ring(40, (0, 300))  # centred on the left edge of the window
    strips = visible_strips(left, right, SIZE)
    assert len(strips) == 1
    assert len(strips[0]) < 2 * 40  # roughly the right half of the ring
    assert np.all(strips[0][:, 0] > -60)


def test_each_visit_to_the_window_is_its_own_strip() -> None:
    # A thin window across the middle of the ring sees its left and right sides. The right side
    # wraps past sample 0 (at angle 0), and must still come out as one strip, not two.
    left, right = ring(40, (400, 10))
    strips = visible_strips(left, right, (800, 20))
    assert len(strips) == 2
    middles = sorted(float(strip[:, 0].mean()) for strip in strips)
    assert middles[0] < 400 < middles[1]


def test_a_road_off_screen_has_no_strips() -> None:
    assert visible_strips(*ring(40, (5000, 5000)), SIZE) == []
