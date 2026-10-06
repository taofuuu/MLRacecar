"""Tests for mlracecar.editor.controller: mouse and keyboard input, without a window."""

import numpy as np
import pytest

from mlracecar.editor.controller import (
    DRAG_THRESHOLD,
    PAN_STEP,
    WIDTH_STEP,
    ZOOM_STEP,
    Button,
    EditorController,
)
from mlracecar.editor.draft import DEFAULT_WIDTH, Point, TrackDraft
from mlracecar.render.camera import Camera

# 1 px = 0.5 m, and the world origin is at the window's centre pixel (400, 300).
CAMERA = Camera(center=(0.0, 0.0), scale=2.0, size=(800, 600))

type Pixel = tuple[float, float]


def pixel_of(point: Point, camera: Camera = CAMERA) -> Pixel:
    x, y = camera.to_screen(point)
    return float(x), float(y)


def click(editor: EditorController, pixel: Pixel, button: Button = Button.LEFT, **kw: bool) -> None:
    editor.press(pixel, button, **kw)
    editor.release(pixel, button)


def drag(editor: EditorController, start: Pixel, end: Pixel, button: Button = Button.LEFT) -> None:
    editor.press(start, button)
    for step in np.linspace(0, 1, 5)[1:]:
        editor.move((start[0] + (end[0] - start[0]) * step, start[1] + (end[1] - start[1]) * step))
    editor.release(end, button)


def square_editor(**kw: bool) -> EditorController:
    """Four points at (+-100, +-100) m, i.e. 200 px either side of the window's centre."""
    points = ((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0))
    return EditorController(TrackDraft(points=points, widths=(12.0,) * 4), CAMERA, **kw)


# --------------------------------------------------------------------------- #
# Drawing a track
# --------------------------------------------------------------------------- #


def test_clicking_on_empty_ground_adds_points_in_order() -> None:
    editor = EditorController(TrackDraft(), CAMERA)
    for pixel in [(600, 300), (400, 100), (200, 300), (400, 500)]:
        click(editor, pixel)
    assert editor.draft.points == ((100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0))
    assert editor.draft.widths == (DEFAULT_WIDTH,) * 4
    assert editor.selected == 3  # the newest point
    assert editor.draft.is_raceable


def test_holding_the_click_places_the_new_point_where_the_mouse_is_let_go() -> None:
    editor = EditorController(TrackDraft(), CAMERA)
    drag(editor, (600, 300), (620, 280))
    assert editor.draft.points == ((110.0, 10.0),)


def test_shift_click_inserts_into_the_nearest_stretch() -> None:
    editor = square_editor()
    click(editor, (400, 96), shift=True)  # just outside the top side, between points 0 and 1
    assert len(editor.draft.points) == 5
    assert editor.draft.points[1] == (0.0, 102.0)
    assert editor.selected == 1


def test_shift_click_with_too_few_points_appends() -> None:
    editor = EditorController(TrackDraft(), CAMERA)
    click(editor, (500, 300), shift=True)
    assert editor.draft.points == ((50.0, 0.0),)
    assert editor.selected == 0


# --------------------------------------------------------------------------- #
# Selecting, moving, and deleting
# --------------------------------------------------------------------------- #


def test_clicking_a_point_selects_it_without_moving_or_adding() -> None:
    editor = square_editor()
    before = editor.draft
    click(editor, (601, 99))  # within grabbing distance of point 0 at (600, 100)
    assert editor.selected == 0
    assert editor.draft == before


def test_a_wobble_smaller_than_the_drag_threshold_does_not_move_the_point() -> None:
    editor = square_editor()
    before = editor.draft
    drag(editor, (600, 100), (600 + DRAG_THRESHOLD - 1, 100))
    assert editor.draft == before


def test_dragging_a_point_moves_it_by_the_mouse_movement() -> None:
    editor = square_editor()
    drag(editor, (604, 104), (644, 64))  # grabbed off-centre, so it must not jump to the mouse
    assert editor.draft.points[0] == (120.0, 120.0)
    assert editor.selected == 0


def test_snapping_puts_points_on_grid_crossings() -> None:
    editor = EditorController(TrackDraft(), CAMERA, snap=True)  # the grid is 10 m at this zoom
    click(editor, (613, 291))  # (106.5, 4.5) m
    assert editor.draft.points == ((110.0, 0.0),)
    drag(editor, (620, 300), (566, 244))  # to (83, 28) m
    assert editor.draft.points == ((80.0, 30.0),)


def test_right_click_deletes_a_point_and_keeps_the_selection_on_the_same_point() -> None:
    editor = square_editor()
    editor.selected = 2
    click(editor, (600, 100), Button.RIGHT)  # delete point 0
    assert len(editor.draft.points) == 3
    assert editor.selected == 1
    assert editor.draft.points[editor.selected] == (-100.0, -100.0)


def test_deleting_the_selected_point_clears_the_selection() -> None:
    editor = square_editor()
    click(editor, (200, 500))
    editor.delete_selected()
    assert (-100.0, -100.0) not in editor.draft.points
    assert editor.selected is None


def test_deleting_after_the_selection_leaves_it_alone() -> None:
    editor = square_editor()
    editor.selected = 0
    click(editor, (600, 500), Button.RIGHT)
    assert editor.selected == 0


def test_keys_that_need_a_selection_do_nothing_without_one() -> None:
    editor = square_editor()
    before = editor.draft
    editor.delete_selected()
    editor.change_selected_width(5.0)
    editor.start_at_selected()
    editor.scroll(1, shift=True)  # the cursor is in the middle, over no point
    assert editor.draft == before


def test_escape_clears_the_selection() -> None:
    editor = square_editor()
    click(editor, (600, 100))
    editor.deselect()
    assert editor.selected is None


def test_hovered_is_the_point_under_the_cursor() -> None:
    editor = square_editor()
    editor.move((205, 495))
    assert editor.hovered == 2
    editor.move((250, 495))
    assert editor.hovered is None


# --------------------------------------------------------------------------- #
# Widths, direction, and the start line
# --------------------------------------------------------------------------- #


def test_shift_wheel_over_a_point_changes_its_width_and_selects_it() -> None:
    editor = square_editor()
    editor.move((200, 100))
    editor.scroll(2, shift=True)
    assert editor.draft.widths[1] == 12.0 + 2 * WIDTH_STEP
    assert editor.selected == 1


def test_shift_wheel_away_from_points_changes_the_selected_width() -> None:
    editor = square_editor()
    click(editor, (200, 500))
    editor.move((400, 300))
    editor.scroll(-3, shift=True)
    assert editor.draft.widths[2] == 12.0 - 3 * WIDTH_STEP


def test_bracket_keys_change_the_selected_width() -> None:
    editor = square_editor()
    click(editor, (600, 500))
    editor.change_selected_width(-WIDTH_STEP)
    assert editor.draft.widths[3] == 12.0 - WIDTH_STEP


def test_reversing_keeps_the_selection_on_the_same_point() -> None:
    editor = square_editor()
    click(editor, (200, 100))
    selected_point = editor.draft.points[editor.selected or 0]
    editor.reverse()
    assert editor.draft.points[editor.selected or 0] == selected_point
    assert editor.draft.points[1] == (100.0, -100.0)


def test_reversing_without_a_selection() -> None:
    editor = square_editor()
    editor.reverse()
    assert editor.selected is None
    assert editor.draft.points[1] == (100.0, -100.0)


def test_the_start_line_moves_to_the_selected_point() -> None:
    editor = square_editor()
    click(editor, (200, 500))
    editor.start_at_selected()
    assert editor.draft.points[0] == (-100.0, -100.0)
    assert editor.selected == 0


# --------------------------------------------------------------------------- #
# The view
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("button", [Button.RIGHT, Button.MIDDLE])
def test_dragging_empty_ground_pans(button: Button) -> None:
    editor = square_editor()
    drag(editor, (400, 300), (450, 250), button)
    assert editor.camera.center == (-25.0, -25.0)
    assert len(editor.draft.points) == 4


def test_middle_drag_on_a_point_pans_instead_of_moving_it() -> None:
    editor = square_editor()
    before = editor.draft
    drag(editor, (600, 100), (650, 100), Button.MIDDLE)
    assert editor.draft == before
    assert editor.camera.center == (-25.0, 0.0)


def test_releasing_a_different_button_does_not_end_the_pan() -> None:
    editor = square_editor()
    editor.press((400, 300), Button.RIGHT)
    editor.release((400, 300), Button.LEFT)
    editor.move((420, 300))
    assert editor.camera.center == (-10.0, 0.0)


def test_a_new_press_ends_a_gesture_whose_release_was_lost() -> None:
    editor = square_editor()
    editor.press((400, 300), Button.RIGHT)  # its release never arrives
    click(editor, (400, 300))
    editor.move((450, 300))
    assert editor.camera.center == (0.0, 0.0)
    assert len(editor.draft.points) == 5


def test_the_wheel_zooms_around_the_cursor() -> None:
    editor = square_editor()
    editor.move((600, 100))
    editor.scroll(1)
    assert editor.camera.scale == pytest.approx(2.0 * ZOOM_STEP)
    np.testing.assert_allclose(editor.camera.to_screen((100.0, 100.0)), (600, 100))


def test_keyboard_zoom_and_pan() -> None:
    editor = square_editor()
    editor.zoom(-1)
    assert editor.camera.scale == pytest.approx(2.0 / ZOOM_STEP)
    assert editor.camera.center == (0.0, 0.0)
    editor.pan(PAN_STEP, 0)
    assert editor.camera.center[0] < 0


def test_fit_shows_the_whole_road() -> None:
    editor = square_editor()
    editor.fit()
    track = editor.draft.track
    assert track is not None
    screen = editor.camera.to_screen(np.concatenate([track.left, track.right]))
    assert np.all((screen >= 0) & (screen <= (800, 600)))


def test_fit_before_there_is_a_track_shows_the_points() -> None:
    editor = EditorController(
        TrackDraft(points=((0.0, 0.0), (300.0, 0.0)), widths=(12.0, 12.0)), CAMERA
    )
    editor.fit()
    assert editor.camera.center == (150.0, 0.0)


def test_snap_toggles_and_the_window_can_resize() -> None:
    editor = square_editor()
    editor.toggle_snap()
    assert editor.snap
    editor.resize((1000, 700))
    assert editor.camera.size == (1000, 700)
    assert editor.camera.center == (0.0, 0.0)
    assert editor.grid_step == 10.0
