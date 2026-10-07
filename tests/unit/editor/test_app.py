"""Tests for mlracecar.editor.app: the pygame window, run off-screen (see tests/conftest.py)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pygame
import pytest

import mlracecar.editor.app as app
from executors import InlineExecutor
from mlracecar.core.track.validation import IssueCode
from mlracecar.editor.app import (
    HELP_LINES,
    SCREEN_SHARE,
    SHORTCUTS,
    WINDOW_SIZE,
    EditorWindow,
    run_editor,
)
from mlracecar.editor.checks import BackgroundChecks
from mlracecar.editor.controller import WIDTH_STEP, ZOOM_STEP
from mlracecar.editor.document import TrackDocument
from mlracecar.editor.draft import TrackDraft
from mlracecar.editor.view import ERROR, READY, WARNING, Message
from mlracecar.io.track_file import read_track_file

SAMPLES = Path(__file__).parents[3] / "tracks"
SQUARE = TrackDraft(
    points=((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0)),
    widths=(12.0,) * 4,
    name="Square",
)


@pytest.fixture(autouse=True)
def close_window() -> Iterator[None]:
    yield
    if pygame.display.get_init():
        pygame.key.set_mods(0)  # tests that hold shift let go of it
    pygame.quit()


def open_window(
    draft: TrackDraft = SQUARE, path: Path | None = None, size: tuple[int, int] | None = (800, 600)
) -> EditorWindow:
    """A window whose track checks run immediately, so tests don't wait for a thread."""
    return EditorWindow(
        TrackDocument(path, draft), size=size, checks=BackgroundChecks(InlineExecutor())
    )


def key(code: int, *, ctrl: bool = False, shift: bool = False) -> pygame.event.Event:
    mod = (pygame.KMOD_CTRL if ctrl else 0) | (pygame.KMOD_SHIFT if shift else 0)
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=mod)


def mouse(kind: int, pos: tuple[int, int], button: int = 1) -> pygame.event.Event:
    return pygame.event.Event(kind, pos=pos, button=button)


def wheel(notches: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=notches, precise_x=0.0, precise_y=notches)


def quit_event() -> pygame.event.Event:
    return pygame.event.Event(pygame.QUIT)


def on_screen(window: EditorWindow, point: tuple[float, float]) -> tuple[int, int]:
    x, y = window.controller.camera.to_screen(point)
    return round(x), round(y)


def click(window: EditorWindow, pos: tuple[int, int], button: int = 1) -> None:
    window.handle(mouse(pygame.MOUSEBUTTONDOWN, pos, button))
    window.handle(mouse(pygame.MOUSEBUTTONUP, pos, button))


def type_text(window: EditorWindow, text: str) -> None:
    """Clear the dialog's text field, type ``text``, and press Enter."""
    assert window.dialog is not None
    for _ in window.dialog.text or "":
        window.handle(key(pygame.K_BACKSPACE))
    window.handle(pygame.event.Event(pygame.TEXTINPUT, text=text))
    window.handle(key(pygame.K_RETURN))


def edit(window: EditorWindow) -> None:
    """Make a change, so there's something unsaved."""
    window.controller.draft = window.controller.draft.change_width(0, 1.0)


# --------------------------------------------------------------------------- #
# Opening the window
# --------------------------------------------------------------------------- #


def test_the_window_fits_on_the_screen() -> None:
    window = open_window(size=None)  # the off-screen driver reports a 1024 x 768 screen
    assert window.controller.camera.size == (int(1024 * SCREEN_SHARE), int(768 * SCREEN_SHARE))
    screen = pygame.display.get_surface()
    assert screen is not None
    assert screen.get_size() == window.controller.camera.size


def test_on_a_big_screen_the_window_keeps_its_usual_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pygame.display, "get_desktop_sizes", lambda: [(3840, 2160)])
    assert open_window(size=None).controller.camera.size == WINDOW_SIZE


def test_a_new_track_opens_with_the_help_showing() -> None:
    window = open_window(TrackDraft())
    assert window.show_help
    window.draw()
    assert pygame.display.get_caption()[0] == "Untitled track - Track editor"


def test_an_existing_track_opens_fitted_to_the_window_without_help() -> None:
    document = TrackDocument.open(SAMPLES / "gp-circuit.json")
    window = EditorWindow(document, checks=BackgroundChecks(InlineExecutor()))
    assert not window.show_help
    track = document.saved.track
    assert track is not None
    low, high = window.controller.camera.visible_area()
    assert (low <= track.left.min(axis=0)).all()
    assert (high >= track.left.max(axis=0)).all()


def test_the_title_shows_the_file_and_a_star_for_unsaved_changes(tmp_path: Path) -> None:
    window = open_window(path=tmp_path / "square.json")
    window.draw()
    assert pygame.display.get_caption()[0] == "Square (square.json) - Track editor"
    edit(window)
    window.draw()
    assert pygame.display.get_caption()[0] == "Square (square.json) * - Track editor"


# --------------------------------------------------------------------------- #
# Mouse and keys
# --------------------------------------------------------------------------- #


def test_clicks_add_points_where_the_mouse_is() -> None:
    window = open_window(TrackDraft())  # 2 px per metre, origin in the middle
    window.handle(mouse(pygame.MOUSEMOTION, (600, 300)))
    click(window, (600, 300))
    click(window, (400, 100))
    assert window.controller.draft.points == ((100.0, 0.0), (0.0, 100.0))


def test_a_click_on_the_road_inserts_and_shift_click_appends() -> None:
    window = open_window()
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
    window = open_window(TrackDraft())
    click(window, (300, 300), button=4)  # pygame also sends the wheel as buttons 4 and 5
    assert window.controller.draft.points == ()


def test_the_wheel_zooms_and_shift_wheel_widens() -> None:
    window = open_window()
    scale = window.controller.camera.scale
    window.handle(wheel(1))
    assert window.controller.camera.scale == pytest.approx(scale * ZOOM_STEP)

    window.handle(mouse(pygame.MOUSEMOTION, on_screen(window, SQUARE.points[2])))
    pygame.key.set_mods(pygame.KMOD_SHIFT)
    window.handle(wheel(2))
    assert window.controller.draft.widths[2] == 12.0 + 2 * WIDTH_STEP


def test_keys_do_what_the_help_says() -> None:
    window = open_window()
    editor = window.controller
    center = editor.camera.center
    window.handle(key(pygame.K_LEFT))
    assert editor.camera.center[0] < center[0]  # the view moves left
    window.handle(key(pygame.K_EQUALS, shift=True))  # "+" on most keyboards
    assert editor.camera.scale > 2.0
    window.handle(key(pygame.K_g))
    assert editor.snap
    window.handle(key(pygame.K_h))
    assert window.show_help
    window.handle(key(pygame.K_i))
    assert not window.show_issue_list

    click(window, on_screen(window, SQUARE.points[2]))
    window.handle(key(pygame.K_RIGHTBRACKET))
    assert editor.draft.widths[2] == 12.0 + WIDTH_STEP
    window.handle(key(pygame.K_s))
    assert editor.draft.points[0] == SQUARE.points[2]
    window.handle(key(pygame.K_DELETE))
    assert len(editor.draft.points) == 3


CHORDS = [chord for shortcut in SHORTCUTS for chord in shortcut.actions]


@pytest.mark.parametrize("chord", CHORDS, ids=lambda chord: str(chord))
def test_every_shortcut_runs(chord: app.Chord) -> None:
    window = open_window()
    window.controller.selected = 1
    window.handle(key(chord.key, ctrl=chord.ctrl, shift=chord.shift))  # no errors


def test_the_help_lists_every_shortcut_and_no_key_does_two_things() -> None:
    assert len(CHORDS) == len(set(CHORDS))
    labels = [label for label, _ in HELP_LINES]
    assert all(shortcut.label in labels for shortcut in SHORTCUTS)
    assert "Hold C + wheel" in labels
    assert all(chord.key != pygame.K_c for chord in CHORDS)  # C is for rounding


def test_ctrl_s_is_not_s() -> None:
    window = open_window()
    window.controller.selected = 2
    window.handle(key(pygame.K_s, ctrl=True))
    assert window.controller.draft.points[0] == SQUARE.points[0]  # the start line didn't move
    assert window.dialog is not None  # it asks where to save instead


def test_unknown_keys_are_ignored() -> None:
    window = open_window()
    window.handle(key(pygame.K_F12))
    assert window.controller.draft == SQUARE


def test_resizing_the_window_resizes_the_view() -> None:
    window = open_window(TrackDraft())
    window.handle(pygame.event.Event(pygame.VIDEORESIZE, size=(900, 700), w=900, h=700))
    assert window.controller.camera.size == (900, 700)


# --------------------------------------------------------------------------- #
# Rounding a corner
# --------------------------------------------------------------------------- #


def key_up(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYUP, key=code, mod=0)


def test_holding_c_and_turning_the_wheel_rounds_a_corner() -> None:
    window = open_window()
    window.handle(mouse(pygame.MOUSEMOTION, on_screen(window, SQUARE.points[0])))
    for _ in range(3):  # the key repeats while it's held
        window.handle(key(pygame.K_c))
    window.handle(wheel(2))
    window.handle(wheel(1))
    rounding = window.controller.rounding
    assert rounding is not None
    assert rounding.radius == 33.0
    assert window.controller.camera.scale == open_window().controller.camera.scale  # no zoom
    window.handle(key_up(pygame.K_c))
    assert window.controller.rounding is None
    assert len(window.controller.draft.points) > 4
    window.handle(key(pygame.K_z, ctrl=True))
    assert window.controller.draft == SQUARE


def test_esc_while_rounding_puts_the_corner_back_and_keeps_the_selection() -> None:
    window = open_window()
    window.handle(mouse(pygame.MOUSEMOTION, on_screen(window, SQUARE.points[0])))
    window.handle(key(pygame.K_c))
    window.handle(key(pygame.K_ESCAPE))
    assert window.controller.draft == SQUARE
    assert window.controller.selected == 0
    window.handle(key_up(pygame.K_c))
    assert window.controller.draft == SQUARE


def test_c_away_from_a_corner_says_what_to_do() -> None:
    window = open_window()
    window.handle(mouse(pygame.MOUSEMOTION, (400, 300)))
    window.handle(key(pygame.K_c))
    assert window.message == Message("Point at a corner, or select one, to round it.", WARNING)
    window.handle(key(pygame.K_c, ctrl=True))  # Ctrl+C isn't rounding
    assert window.controller.rounding is None


# --------------------------------------------------------------------------- #
# Undo and redo
# --------------------------------------------------------------------------- #


def test_ctrl_z_undoes_and_ctrl_y_or_ctrl_shift_z_redoes(tmp_path: Path) -> None:
    window = open_window(path=tmp_path / "square.json")
    click(window, on_screen(window, (0.0, 0.0)))  # adds a point
    added = window.controller.draft
    window.draw()
    assert pygame.display.get_caption()[0].endswith(" * - Track editor")
    window.handle(key(pygame.K_z, ctrl=True))
    assert window.controller.draft == SQUARE
    window.draw()
    assert " * " not in pygame.display.get_caption()[0]  # back to what's saved
    window.handle(key(pygame.K_y, ctrl=True))
    assert window.controller.draft == added
    window.handle(key(pygame.K_z, ctrl=True))
    window.handle(key(pygame.K_z, ctrl=True, shift=True))
    assert window.controller.draft == added


def test_with_nothing_to_undo_or_redo_the_window_says_so() -> None:
    window = open_window()
    window.handle(key(pygame.K_z, ctrl=True))
    assert window.message == Message("Nothing to undo.", WARNING)
    window.handle(key(pygame.K_y, ctrl=True))
    assert window.message == Message("Nothing to redo.", WARNING)


def test_saving_is_not_a_step_to_undo(tmp_path: Path) -> None:
    window = open_window(TrackDraft(points=SQUARE.points, widths=SQUARE.widths))
    click(window, on_screen(window, (0.0, 0.0)))
    window.save_as()
    type_text(window, (tmp_path / "named.json").as_posix())
    saved = window.controller.draft
    assert saved.name == "named"
    window.handle(key(pygame.K_z, ctrl=True))
    assert window.controller.draft.points == SQUARE.points  # the click, not the naming
    window.handle(key(pygame.K_y, ctrl=True))
    assert window.controller.draft == saved


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #


def test_ctrl_s_saves_to_the_open_file(tmp_path: Path) -> None:
    window = open_window(path=tmp_path / "square.json")
    edit(window)
    window.handle(key(pygame.K_s, ctrl=True))
    assert read_track_file(tmp_path / "square.json").widths[0] == 13.0
    assert not window.document.is_modified(window.controller.draft)
    assert window.message is not None
    assert window.message.text == f"Saved {tmp_path / 'square.json'}."
    assert window.message.color == READY


def test_saving_a_new_track_asks_where_and_names_it_after_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    window = open_window(TrackDraft(points=SQUARE.points, widths=SQUARE.widths))
    window.handle(key(pygame.K_s, ctrl=True))
    assert window.dialog is not None
    assert window.dialog.text == "untitled-track.json"
    type_text(window, "my-oval")  # ".json" is added
    assert (tmp_path / "my-oval.json").exists()
    assert window.controller.draft.name == "my-oval"
    assert window.document.path == Path("my-oval.json")
    assert window.dialog is None


def test_save_as_offers_the_current_file_with_forward_slashes(tmp_path: Path) -> None:
    window = open_window(path=tmp_path / "square.json")
    window.handle(key(pygame.K_s, ctrl=True, shift=True))
    assert window.dialog is not None
    assert window.dialog.text == (tmp_path / "square.json").as_posix()


def test_saving_over_another_file_asks_first(tmp_path: Path) -> None:
    other = tmp_path / "other.json"
    other.write_text("keep me", encoding="utf-8")
    window = open_window(path=tmp_path / "square.json")
    window.handle(key(pygame.K_s, ctrl=True, shift=True))
    type_text(window, other.as_posix())
    assert window.dialog is not None
    assert "already exists" in window.dialog.title
    window.handle(key(pygame.K_ESCAPE))
    assert other.read_text(encoding="utf-8") == "keep me"

    window.handle(key(pygame.K_s, ctrl=True, shift=True))
    type_text(window, other.as_posix())
    window.handle(key(pygame.K_RETURN))
    assert read_track_file(other).name == "Square"


def test_saving_a_track_with_errors_says_it_cannot_be_raced_yet(tmp_path: Path) -> None:
    window = open_window(SQUARE.set_width(1, 3.0), path=tmp_path / "narrow.json")
    window.handle(key(pygame.K_s, ctrl=True))
    assert (tmp_path / "narrow.json").exists()
    assert window.message is not None
    assert "but with 1 error: can't race yet" in window.message.text
    assert window.message.color == WARNING


def test_the_message_counts_several_errors(tmp_path: Path) -> None:
    narrow = SQUARE.set_width(0, 3.0).set_width(2, 3.0)
    window = open_window(narrow, path=tmp_path / "narrow.json")
    window.handle(key(pygame.K_s, ctrl=True))
    assert window.message is not None
    assert "with 2 errors" in window.message.text


def test_a_failed_save_says_why(tmp_path: Path) -> None:
    window = open_window(TrackDraft().append_point((0.0, 0.0)), path=tmp_path / "dot.json")
    window.handle(key(pygame.K_s, ctrl=True))
    assert window.message is not None
    assert window.message.text.startswith("Can't save:")
    assert "at least 3 control points" in window.message.text
    assert window.message.color == ERROR
    assert not (tmp_path / "dot.json").exists()


def test_an_empty_file_name_is_not_saved(tmp_path: Path) -> None:
    window = open_window()
    window.handle(key(pygame.K_s, ctrl=True))
    type_text(window, "   ")
    assert window.message is not None
    assert window.message.text == "Type a file name to save to."
    assert window.document.path is None


def test_a_dialog_takes_the_keyboard_until_answered() -> None:
    window = open_window()
    window.handle(key(pygame.K_F2))
    window.handle(key(pygame.K_g))  # would turn on grid snap without the dialog
    click(window, (100, 100))  # the mouse is ignored too
    assert not window.controller.snap
    assert window.controller.draft == SQUARE
    window.handle(key(pygame.K_ESCAPE))
    assert window.dialog is None


def test_f2_renames_the_track() -> None:
    window = open_window()
    window.handle(key(pygame.K_F2))
    assert window.dialog is not None
    assert window.dialog.text == "Square"
    type_text(window, "Silverstone-ish")
    assert window.controller.draft.name == "Silverstone-ish"
    window.handle(key(pygame.K_z, ctrl=True))
    assert window.controller.draft.name == "Square"


# --------------------------------------------------------------------------- #
# Closing
# --------------------------------------------------------------------------- #


def test_closing_without_changes_closes_at_once() -> None:
    window = open_window()
    window.handle(quit_event())
    assert not window.running


def test_closing_with_unsaved_changes_asks_first() -> None:
    window = open_window()
    edit(window)
    window.handle(quit_event())
    assert window.running
    assert window.dialog is not None
    window.handle(key(pygame.K_ESCAPE))  # keep editing
    assert window.running
    window.handle(quit_event())
    window.handle(key(pygame.K_q))  # close without saving
    assert not window.running


def test_closing_can_save_first(tmp_path: Path) -> None:
    window = open_window(path=tmp_path / "square.json")
    edit(window)
    window.handle(quit_event())
    window.handle(key(pygame.K_RETURN))
    assert not window.running
    assert read_track_file(tmp_path / "square.json").widths[0] == 13.0


def test_closing_a_new_track_asks_for_a_file_then_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    window = open_window(path=None)
    edit(window)
    window.handle(quit_event())
    window.handle(key(pygame.K_RETURN))
    assert is_running(window)  # asking for a file name now
    type_text(window, "square")
    assert not is_running(window)
    assert (tmp_path / "square.json").exists()


def test_closing_stays_open_if_saving_fails(tmp_path: Path) -> None:
    window = open_window(TrackDraft(), path=tmp_path / "empty.json")
    window.controller.draft = TrackDraft().append_point((0.0, 0.0))
    window.handle(quit_event())
    window.handle(key(pygame.K_RETURN))
    assert window.running
    assert window.message is not None
    assert window.message.color == ERROR


# --------------------------------------------------------------------------- #
# Track checks
# --------------------------------------------------------------------------- #


def test_problems_found_by_the_checks_are_shown() -> None:
    window = open_window(SQUARE.set_width(1, 3.0))
    assert window.overlays().checking  # nothing checked yet
    window.checks.check(window.controller.draft)
    window.checks.poll()
    overlays = window.overlays()
    assert not overlays.checking
    assert [issue.code for issue in overlays.issues] == [IssueCode.TOO_NARROW]


def test_results_for_an_older_draft_show_until_points_are_added_or_removed() -> None:
    window = open_window(SQUARE.set_width(1, 3.0))
    window.checks.check(window.controller.draft)
    window.checks.poll()
    window.controller.draft = window.controller.draft.move_point(0, (110.0, 110.0))
    assert window.overlays().issues  # same points, slightly moved: still meaningful
    window.controller.draft = window.controller.draft.delete_point(3)
    assert window.overlays().checking  # point numbers changed: wait for the new check
    assert not window.overlays().issues


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_run_draws_until_the_window_is_closed_and_returns_the_draft() -> None:
    window = open_window(TrackDraft())
    for pos in [(600, 300), (400, 100), (200, 300), (400, 500)]:
        pygame.event.post(mouse(pygame.MOUSEBUTTONDOWN, pos))
        pygame.event.post(mouse(pygame.MOUSEBUTTONUP, pos))
    pygame.event.post(quit_event())
    pygame.event.post(key(pygame.K_q))  # close without saving
    draft = window.run()
    assert len(draft.points) == 4
    assert draft.is_raceable
    assert not pygame.display.get_init()


def test_a_new_track_can_be_drawn_and_saved_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ticket #14's goal, as a user would do it: click round a loop, press Ctrl+S, name the file,
    # close the window. Every step goes through the real event loop.
    monkeypatch.chdir(tmp_path)
    window = open_window(TrackDraft())  # 2 px per metre, origin in the middle of 800 x 600
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    for angle in angles:
        pos = (round(400 + 260 * np.cos(angle)), round(300 - 160 * np.sin(angle)))
        pygame.event.post(mouse(pygame.MOUSEBUTTONDOWN, pos))
        pygame.event.post(mouse(pygame.MOUSEBUTTONUP, pos))
    pygame.event.post(key(pygame.K_s, ctrl=True))
    for _ in "untitled-track.json":
        pygame.event.post(key(pygame.K_BACKSPACE))
    pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text="tracks/first-oval"))
    pygame.event.post(key(pygame.K_RETURN))
    pygame.event.post(quit_event())  # saved, so it closes without asking
    window.run()

    saved = read_track_file(tmp_path / "tracks" / "first-oval.json")
    assert saved.name == "first-oval"
    assert len(saved.control_points) == 10
    assert TrackDraft.from_track_file(saved).issues == []  # a valid, raceable track


def test_an_idle_window_does_not_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    window = open_window()
    frames: list[None] = []
    monkeypatch.setattr(window, "draw", lambda: frames.append(None))
    pygame.event.clear()
    pygame.time.set_timer(pygame.QUIT, 100, loops=1)  # close the window in 0.1 s
    window.run()
    assert len(frames) == 2  # the first frame, and again when the track check finished


def test_messages_go_away_after_a_while(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "MESSAGE_SECONDS", 0.02)
    window = open_window(path=tmp_path / "square.json")
    window.handle(key(pygame.K_s, ctrl=True))
    assert window.message is not None
    pygame.event.clear()
    pygame.time.set_timer(pygame.QUIT, 100, loops=1)
    window.run()
    assert message_of(window) is None


def message_of(window: EditorWindow) -> Message | None:
    return window.message  # mypy can't see `run` or `handle` changing it, so read it fresh


def is_running(window: EditorWindow) -> bool:
    return window.running  # likewise


def test_run_editor_starts_from_an_empty_track() -> None:
    pygame.display.init()
    pygame.event.post(quit_event())
    assert run_editor() == TrackDraft()


def test_run_editor_opens_the_given_document() -> None:
    pygame.display.init()
    pygame.event.post(quit_event())
    assert run_editor(TrackDocument(None, SQUARE)) == SQUARE
