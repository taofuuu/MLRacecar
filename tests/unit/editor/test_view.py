"""Tests for mlracecar.editor.view: what the editor draws, checked on images in memory."""

from collections.abc import Iterator

import numpy as np
import pygame
import pytest

from mlracecar.editor.controller import EditorController
from mlracecar.editor.draft import Point, TrackDraft
from mlracecar.editor.view import (
    HOVERED,
    OUTLINE,
    PANEL,
    POINT,
    SELECTED,
    START_POINT,
    STATUS_HEIGHT,
    EditorView,
)
from mlracecar.render.camera import Camera
from mlracecar.render.drawing import BACKGROUND, ROAD

SIZE = (800, 600)
CAMERA = Camera(center=(0.0, 0.0), scale=2.0, size=SIZE)
SQUARE = ((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0))


@pytest.fixture(autouse=True)
def fonts() -> Iterator[None]:
    pygame.font.init()
    yield
    pygame.font.quit()


def render(
    editor: EditorController, help_lines: tuple[tuple[str, str], ...] = ()
) -> pygame.Surface:
    surface = pygame.Surface(SIZE)
    EditorView().draw(surface, editor, help_lines)
    return surface


def color_at(surface: pygame.Surface, point: Point) -> tuple[int, int, int]:
    x, y = np.round(CAMERA.to_screen(point)).astype(int)
    r, g, b, _ = surface.get_at((int(x), int(y)))
    return r, g, b


def square(*, snap: bool = False) -> EditorController:
    return EditorController(TrackDraft(points=SQUARE, widths=(12.0,) * 4), CAMERA, snap=snap)


def test_a_track_is_drawn_with_its_points() -> None:
    editor = square()
    surface = render(editor)
    track = editor.draft.track
    assert track is not None
    quarter = track.centerline.points[len(track.centerline.points) // 8]  # between two points
    assert color_at(surface, (float(quarter[0]), float(quarter[1]))) == ROAD
    assert color_at(surface, SQUARE[0]) == START_POINT
    assert color_at(surface, SQUARE[1]) == POINT
    assert color_at(surface, (0.0, 0.0)) != ROAD  # the infield


def test_until_the_points_make_a_track_they_are_joined_in_order() -> None:
    editor = EditorController(TrackDraft(points=SQUARE[:2], widths=(12.0, 12.0)), CAMERA)
    surface = render(editor)
    assert color_at(surface, (0.0, 100.0)) == OUTLINE
    assert color_at(surface, (0.0, 0.0)) != OUTLINE


def test_a_single_point_is_just_a_dot() -> None:
    editor = EditorController(TrackDraft(points=SQUARE[:1], widths=(12.0,)), CAMERA)
    assert color_at(render(editor), SQUARE[0]) == START_POINT


def test_selected_and_hovered_points_stand_out() -> None:
    editor = square()
    editor.selected = 2
    x, y = CAMERA.to_screen(SQUARE[1])
    editor.move((float(x), float(y)))
    surface = render(editor)
    assert color_at(surface, SQUARE[1]) == HOVERED
    ring = (SQUARE[2][0] + 4.0, SQUARE[2][1])  # 8 px right of the selected point
    assert color_at(surface, ring) == SELECTED


def hint_pixels(editor: EditorController, cursor: Point) -> int:
    x, y = CAMERA.to_screen(cursor)
    editor.move((float(x), float(y)))
    area = render(editor).subsurface((round(x) - 8, round(y) - 8, 17, 17))
    return int(pygame.mask.from_threshold(area, HOVERED, (1, 1, 1, 255)).count())


def test_a_hollow_dot_shows_where_a_click_would_insert_into_the_road() -> None:
    editor = square()
    track = editor.draft.track
    assert track is not None
    on_road = track.centerline.points[len(track.centerline.points) // 8]
    assert hint_pixels(editor, (float(on_road[0]), float(on_road[1]))) > 0
    assert hint_pixels(editor, (0.0, 0.0)) == 0  # off the road: a click adds after the last point


def test_the_status_bar_runs_along_the_bottom() -> None:
    surface = render(square(snap=True))
    assert surface.get_at((SIZE[0] // 2, SIZE[1] - 2))[:3] == PANEL[:3]
    assert surface.get_at((SIZE[0] // 2, SIZE[1] - STATUS_HEIGHT - 2))[:3] != PANEL[:3]


def test_the_status_bar_describes_the_selected_point() -> None:
    plain, selected = square(), square()
    selected.selected = 1
    # More text in the bar means more lit pixels; the bar is otherwise identical.
    assert lit_pixels_in_status_bar(render(selected)) > lit_pixels_in_status_bar(render(plain))


def lit_pixels_in_status_bar(surface: pygame.Surface) -> int:
    bar = surface.subsurface((0, SIZE[1] - STATUS_HEIGHT, SIZE[0], STATUS_HEIGHT))
    return int(pygame.mask.from_threshold(bar, (226, 232, 240), (60, 60, 60, 255)).count())


def test_the_help_panel_covers_the_middle_of_the_window() -> None:
    editor = EditorController(TrackDraft(), CAMERA)
    without, with_help = render(editor), render(editor, (("Click", "add a point"),))
    middle = (SIZE[0] // 2, SIZE[1] // 2)
    assert without.get_at(middle)[:3] != with_help.get_at(middle)[:3]
    assert without.get_at((5, 5)) == with_help.get_at((5, 5))


def test_an_empty_draft_shows_only_the_grid() -> None:
    surface = render(EditorController(TrackDraft(), CAMERA))
    assert surface.get_at((3, 3))[:3] == BACKGROUND
