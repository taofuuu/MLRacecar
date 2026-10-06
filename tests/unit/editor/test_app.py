"""Tests for mlracecar.editor.app: the pygame window, run off-screen (see tests/conftest.py)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pygame
import pytest

from mlracecar.editor.app import (
    HELP_KEY,
    HELP_LINES,
    SCREEN_SHARE,
    SHORTCUTS,
    WINDOW_SIZE,
    EditorWindow,
    run_editor,
)
from mlracecar.editor.controller import WIDTH_STEP, ZOOM_STEP
from mlracecar.editor.draft import TrackDraft
from mlracecar.io.track_file import read_track_file

SAMPLES = Path(__file__).parents[3] / "tracks"
SQUARE = TrackDraft(
    points=((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0)),
    widths=(12.0,) * 4,
)


@pytest.fixture(autouse=True)
def close_window() -> Iterator[None]:
    yield
    if pygame.display.get_init():
        pygame.key.set_mods(0)  # tests that hold shift let go of it
    pygame.quit()


def key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=0)


def mouse(kind: int, pos: tuple[int, int], button: int = 1) -> pygame.event.Event:
    return pygame.event.Event(kind, pos=pos, button=button)


def wheel(notches: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=notches, precise_x=0.0, precise_y=notches)


def on_screen(window: EditorWindow, point: tuple[float, float]) -> tuple[int, int]:
    x, y = window.controller.camera.to_screen(point)
    return round(x), round(y)


def click(window: EditorWindow, pos: tuple[int, int], button: int = 1) -> None:
    window.handle(mouse(pygame.MOUSEBUTTONDOWN, pos, button))
    window.handle(mouse(pygame.MOUSEBUTTONUP, pos, button))


# --------------------------------------------------------------------------- #
# Opening the window
# --------------------------------------------------------------------------- #


def test_the_window_fits_on_the_screen() -> None:
    window = EditorWindow(TrackDraft())  # the off-screen driver reports a 1024 x 768 screen
    assert window.controller.camera.size == (int(1024 * SCREEN_SHARE), int(768 * SCREEN_SHARE))
    screen = pygame.display.get_surface()
    assert screen is not None
    assert screen.get_size() == window.controller.camera.size


def test_on_a_big_screen_the_window_keeps_its_usual_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pygame.display, "get_desktop_sizes", lambda: [(3840, 2160)])
    assert EditorWindow(TrackDraft()).controller.camera.size == WINDOW_SIZE


def test_a_new_track_opens_with_the_help_showing() -> None:
    window = EditorWindow(TrackDraft())
    assert window.show_help
    assert pygame.display.get_caption()[0] == "Track editor: Untitled track"


def test_an_existing_track_opens_fitted_to_the_window_without_help() -> None:
    draft = TrackDraft.from_track_file(read_track_file(SAMPLES / "gp-circuit.json"))
    window = EditorWindow(draft)
    assert not window.show_help
    track = draft.track
    assert track is not None
    low, high = window.controller.camera.visible_area()
    assert (low <= track.left.min(axis=0)).all()
    assert (high >= track.left.max(axis=0)).all()


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def test_clicks_add_points_where_the_mouse_is() -> None:
    window = EditorWindow(TrackDraft(), size=(800, 600))  # 2 px per metre, origin in the middle
    window.handle(mouse(pygame.MOUSEMOTION, (600, 300)))
    click(window, (600, 300))
    click(window, (400, 100))
    assert window.controller.draft.points == ((100.0, 0.0), (0.0, 100.0))


def test_a_click_on_the_road_inserts_and_shift_click_appends() -> None:
    window = EditorWindow(SQUARE, size=(800, 600))
    track = SQUARE.track
    assert track is not None

    def on_road(after: int) -> tuple[int, int]:
        stretch = np.flatnonzero(track.centerline.piece == after)
        return on_screen(window, tuple(track.centerline.points[stretch[len(stretch) // 2]]))

    click(window, on_road(after=0))
    assert window.controller.selected == 1
    pygame.key.set_mods(pygame.KMOD_SHIFT)
    click(window, on_road(after=2))
    assert window.controller.selected == 5


def test_the_old_wheel_buttons_are_ignored() -> None:
    window = EditorWindow(TrackDraft())
    click(window, (300, 300), button=4)  # pygame also sends the wheel as buttons 4 and 5
    assert window.controller.draft.points == ()


def test_the_wheel_zooms_and_shift_wheel_widens() -> None:
    window = EditorWindow(SQUARE)
    scale = window.controller.camera.scale
    window.handle(wheel(1))
    assert window.controller.camera.scale == pytest.approx(scale * ZOOM_STEP)

    window.handle(mouse(pygame.MOUSEMOTION, on_screen(window, SQUARE.points[2])))
    pygame.key.set_mods(pygame.KMOD_SHIFT)
    window.handle(wheel(2))
    assert window.controller.draft.widths[2] == 12.0 + 2 * WIDTH_STEP


def test_keys_do_what_the_help_says() -> None:
    window = EditorWindow(SQUARE)
    editor = window.controller
    center = editor.camera.center
    window.handle(key(pygame.K_LEFT))
    assert editor.camera.center[0] < center[0]  # the view moves left
    window.handle(key(pygame.K_g))
    assert editor.snap
    window.handle(key(HELP_KEY))
    assert window.show_help

    click(window, on_screen(window, SQUARE.points[2]))
    window.handle(key(pygame.K_RIGHTBRACKET))
    assert editor.draft.widths[2] == 12.0 + WIDTH_STEP
    window.handle(key(pygame.K_s))
    assert editor.draft.points[0] == SQUARE.points[2]
    window.handle(key(pygame.K_DELETE))
    assert len(editor.draft.points) == 3


@pytest.mark.parametrize(
    "code", [code for shortcut in SHORTCUTS for code in shortcut.actions], ids=pygame.key.name
)
def test_every_shortcut_runs(code: int) -> None:
    window = EditorWindow(SQUARE)
    window.controller.selected = 1
    window.handle(key(code))  # no errors, whatever the key does


def test_the_help_lists_every_shortcut_once_and_no_key_does_two_things() -> None:
    codes = [code for shortcut in SHORTCUTS for code in shortcut.actions]
    assert len(codes) == len(set(codes))
    assert HELP_KEY not in codes
    labels = [label for label, _ in HELP_LINES]
    assert all(shortcut.label in labels for shortcut in SHORTCUTS)


def test_unknown_keys_are_ignored() -> None:
    window = EditorWindow(SQUARE)
    window.handle(key(pygame.K_F12))
    assert window.controller.draft == SQUARE


def test_resizing_the_window_resizes_the_view() -> None:
    window = EditorWindow(TrackDraft())
    window.handle(pygame.event.Event(pygame.VIDEORESIZE, size=(900, 700), w=900, h=700))
    assert window.controller.camera.size == (900, 700)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_run_draws_until_the_window_is_closed_and_returns_the_draft() -> None:
    window = EditorWindow(TrackDraft(), size=(800, 600))
    for pos in [(600, 300), (400, 100), (200, 300), (400, 500)]:
        pygame.event.post(mouse(pygame.MOUSEBUTTONDOWN, pos))
        pygame.event.post(mouse(pygame.MOUSEBUTTONUP, pos))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    draft = window.run()
    assert len(draft.points) == 4
    assert draft.is_raceable
    assert not pygame.display.get_init()


def test_an_idle_window_does_not_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    window = EditorWindow(SQUARE)
    frames: list[None] = []
    monkeypatch.setattr(window, "draw", lambda: frames.append(None))
    pygame.event.clear()
    pygame.time.set_timer(pygame.QUIT, 100, loops=1)  # close the window in 0.1 s
    window.run()
    assert len(frames) == 1  # the first frame, then nothing until the window closes


def test_run_editor_starts_from_an_empty_track() -> None:
    pygame.display.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert run_editor() == TrackDraft()


def test_run_editor_opens_the_given_draft() -> None:
    pygame.display.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert run_editor(SQUARE) == SQUARE
