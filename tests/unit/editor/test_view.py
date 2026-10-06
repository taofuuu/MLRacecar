"""Tests for mlracecar.editor.view: what the editor draws, checked on images in memory."""

from collections.abc import Iterator

import numpy as np
import pygame
import pytest

from mlracecar.core.track.validation import (
    ControlPointAt,
    IssueCode,
    Location,
    Severity,
    Spot,
    Stretch,
    ValidationIssue,
)
from mlracecar.editor.controller import EditorController
from mlracecar.editor.draft import Point, TrackDraft
from mlracecar.editor.view import (
    ERROR,
    HOVERED,
    OUTLINE,
    PANEL,
    POINT,
    READY,
    SELECTED,
    START_POINT,
    STATUS_HEIGHT,
    WARNING,
    DialogBox,
    EditorView,
    Message,
    Overlays,
    _stretch_samples,
    issues_under_cursor,
)
from mlracecar.render.camera import Camera
from mlracecar.render.drawing import BACKGROUND, ROAD

SIZE = (800, 600)
CAMERA = Camera(center=(0.0, 0.0), scale=2.0, size=SIZE)
SQUARE = ((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0))
CANVAS = pygame.Rect(0, 0, SIZE[0], SIZE[1] - STATUS_HEIGHT)


@pytest.fixture(autouse=True)
def fonts() -> Iterator[None]:
    pygame.font.init()
    yield
    pygame.font.quit()


def render(editor: EditorController, overlays: Overlays | None = None) -> pygame.Surface:
    surface = pygame.Surface(SIZE)
    EditorView().draw(surface, editor, overlays or Overlays())
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
    without = render(editor)
    with_help = render(editor, Overlays(help_lines=(("Click", "add a point"),)))
    middle = (SIZE[0] // 2, SIZE[1] // 2)
    assert without.get_at(middle)[:3] != with_help.get_at(middle)[:3]
    assert without.get_at((5, 5)) == with_help.get_at((5, 5))


def test_an_empty_draft_shows_only_the_grid() -> None:
    surface = render(EditorController(TrackDraft(), CAMERA))
    assert surface.get_at((3, 3))[:3] == BACKGROUND


# --------------------------------------------------------------------------- #
# Track problems
# --------------------------------------------------------------------------- #


def issue(location: Location, severity: Severity = Severity.ERROR) -> ValidationIssue:
    return ValidationIssue(severity, IssueCode.TOO_NARROW, "Something is wrong here.", location)


def pixel_of(point: Point) -> tuple[int, int]:
    x, y = np.round(CAMERA.to_screen(point)).astype(int)
    return int(x), int(y)


def color_at_pixel(surface: pygame.Surface, pixel: tuple[int, int]) -> tuple[int, int, int]:
    r, g, b, _ = surface.get_at(pixel)
    return r, g, b


def count(surface: pygame.Surface, color: tuple[int, int, int], area: pygame.Rect) -> int:
    return int(pygame.mask.from_threshold(surface.subsurface(area), color, (1, 1, 1, 255)).count())


def centerline_point(editor: EditorController, metres: float) -> Point:
    track = editor.draft.track
    assert track is not None
    x, y = track.pose_at(metres).position
    return float(x), float(y)


def test_a_problem_at_a_point_gets_a_ring_round_it() -> None:
    surface = render(square(), Overlays(issues=[issue(ControlPointAt(1))]))
    x, y = pixel_of(SQUARE[1])
    assert color_at_pixel(surface, (x + 11, y)) == ERROR


def test_a_problem_along_a_stretch_colours_its_road_edges() -> None:
    editor = square()
    track = editor.draft.track
    assert track is not None
    surface = render(editor, Overlays(issues=[issue(Stretch(10.0, 60.0), Severity.WARNING)]))
    inside, outside = track.centerline.arc_length.searchsorted([35.0, 120.0])
    assert color_at(surface, tuple(track.left[inside])) == WARNING
    assert color_at(surface, tuple(track.right[inside])) == WARNING
    assert color_at(surface, tuple(track.left[outside])) != WARNING


def test_a_stretch_too_short_to_see_also_gets_a_ring() -> None:
    editor = square()
    middle = centerline_point(editor, 50.0)
    ahead = centerline_point(editor, 56.0)  # 12 px further along the middle of the road
    short = render(editor, Overlays(issues=[issue(Stretch(49.5, 50.5))]))
    long = render(editor, Overlays(issues=[issue(Stretch(20.0, 80.0))]))
    assert color_at(short, ahead) == ERROR  # on the ring
    assert color_at(long, ahead) == ROAD  # long stretches only colour their edges
    assert color_at(short, middle) == ROAD


def test_a_stretch_through_the_start_line_runs_in_driving_order() -> None:
    track = square().draft.track
    assert track is not None
    samples = _stretch_samples(track, track.length - 5.0, 5.0)
    np.testing.assert_array_equal(np.diff(samples) % len(track.left), 1)
    assert samples[0] > samples[-1]  # it wraps from the end of the lap to the start


def test_a_stretch_shorter_than_the_sample_spacing_still_gets_a_line() -> None:
    track = square().draft.track
    assert track is not None
    samples = _stretch_samples(track, 50.0, 50.1)  # samples are about 0.5 m apart
    assert len(samples) == 2
    assert samples[1] == samples[0] + 1


def test_a_problem_spot_gets_a_ring() -> None:
    surface = render(square(), Overlays(issues=[issue(Spot(0.0, 0.0), Severity.WARNING)]))
    x, y = pixel_of((0.0, 0.0))
    assert color_at_pixel(surface, (x, y - 13)) == WARNING
    assert color_at_pixel(surface, (x, y)) != WARNING


def test_problems_at_points_that_no_longer_exist_are_skipped() -> None:
    surface = render(square(), Overlays(issues=[issue(ControlPointAt(9))], show_issue_list=False))
    assert count(surface, ERROR, CANVAS) == 0  # the status bar says "1 error" in red: skip it


def test_stretches_are_skipped_until_there_is_a_track() -> None:
    editor = EditorController(TrackDraft(points=SQUARE[:2], widths=(12.0, 12.0)), CAMERA)
    surface = render(editor, Overlays(issues=[issue(Stretch(0.0, 10.0))], show_issue_list=False))
    assert count(surface, ERROR, CANVAS) == 0  # the status bar says "1 error" in red: skip it


def test_the_problem_list_sits_in_the_top_left_corner_and_can_be_hidden() -> None:
    issues = [issue(None)] * 8
    corner = pygame.Rect(10, 10, 40, 120)
    shown = render(square(), Overlays(issues=issues))
    hidden = render(square(), Overlays(issues=issues, show_issue_list=False))
    assert count(shown, ERROR, corner) > 0
    assert count(hidden, ERROR, corner) == 0


def test_long_messages_are_shortened_in_the_list() -> None:
    long = ValidationIssue(Severity.WARNING, IssueCode.TIGHT_BEND, "word " * 200, None)
    surface = render(square(), Overlays(issues=[long]))
    # One line per problem: the second row of the panel is the hint, not more of the message.
    assert count(surface, WARNING, pygame.Rect(10, 10, 40, 30)) > 0
    assert count(surface, WARNING, pygame.Rect(10, 40, 40, 60)) == 0


@pytest.mark.parametrize(
    ("location", "cursor", "found"),
    [
        (ControlPointAt(1), SQUARE[1], True),
        (ControlPointAt(1), (-80.0, 100.0), False),
        (ControlPointAt(7), SQUARE[1], False),  # no such point any more
        (Spot(10.0, 10.0), (12.0, 12.0), True),
        (Spot(10.0, 10.0), (30.0, 30.0), False),
        (None, (0.0, 0.0), False),
    ],
)
def test_problems_under_the_cursor(location: Location, cursor: Point, found: bool) -> None:
    editor = square()
    x, y = CAMERA.to_screen(cursor)
    editor.move((float(x), float(y)))
    assert (issues_under_cursor(editor, [issue(location)]) != []) is found


def test_a_stretch_is_under_the_cursor_anywhere_along_it_on_the_road() -> None:
    editor = square()
    problem = issue(Stretch(20.0, 80.0))
    for metres, found in [(50.0, True), (21.0, True), (150.0, False)]:
        x, y = CAMERA.to_screen(centerline_point(editor, metres))
        editor.move((float(x), float(y)))
        assert (issues_under_cursor(editor, [problem]) != []) is found
    editor.move((400.0, 300.0))  # the infield, off the road
    assert issues_under_cursor(editor, [problem]) == []


def test_pointing_at_a_problem_shows_its_message_next_to_the_cursor() -> None:
    editor = square()
    x, y = CAMERA.to_screen(SQUARE[1])
    editor.move((float(x), float(y)))
    tip = pygame.Rect(round(x) + 18, round(y) + 18, 40, 30)
    with_tip = render(editor, Overlays(issues=[issue(ControlPointAt(1))], show_issue_list=False))
    without = render(editor, Overlays(issues=[issue(ControlPointAt(3))], show_issue_list=False))
    assert count(with_tip, ERROR, tip) > 0  # the bullet before the message
    assert count(without, ERROR, tip) == 0


def test_a_long_message_wraps_onto_several_lines_in_the_tooltip() -> None:
    editor = square()
    x, y = CAMERA.to_screen(SQUARE[1])
    editor.move((float(x), float(y)))
    long = ValidationIssue(Severity.ERROR, IssueCode.TOO_NARROW, "word " * 60, ControlPointAt(1))
    tip = pygame.Rect(round(x) + 18, round(y) + 18, 360, 200)
    surface = render(editor, Overlays(issues=[long], show_issue_list=False))
    assert count(surface, ERROR, tip) > 0  # one bullet, for the first line
    lines_of_text = count(surface, (226, 232, 240), tip)
    short = render(editor, Overlays(issues=[issue(ControlPointAt(1))], show_issue_list=False))
    assert lines_of_text > 3 * count(short, (226, 232, 240), tip)


def test_no_tooltip_while_a_dialog_is_open() -> None:
    editor = square()
    x, y = CAMERA.to_screen(SQUARE[1])
    editor.move((float(x), float(y)))
    tip = pygame.Rect(round(x) + 18, round(y) + 18, 40, 30)
    dialog = DialogBox("Save?", "Enter: save")
    hidden = Overlays(issues=[issue(ControlPointAt(1))], show_issue_list=False, dialog=dialog)
    assert count(render(editor, hidden), ERROR, tip) == 0


# --------------------------------------------------------------------------- #
# Status bar verdict, messages, and dialogs
# --------------------------------------------------------------------------- #


def status_bar() -> pygame.Rect:
    return pygame.Rect(SIZE[0] // 2, SIZE[1] - STATUS_HEIGHT, SIZE[0] // 2, STATUS_HEIGHT)


@pytest.mark.parametrize(
    ("issues", "color"),
    [
        ([issue(None), issue(None, Severity.WARNING)], ERROR),
        ([issue(None), issue(None)], ERROR),
        ([issue(None, Severity.WARNING)], WARNING),
        ([issue(None, Severity.WARNING)] * 2, WARNING),
        ([], READY),
    ],
)
def test_the_status_bar_says_whether_the_track_can_be_raced(
    issues: list[ValidationIssue], color: tuple[int, int, int]
) -> None:
    surface = render(square(), Overlays(issues=issues, show_issue_list=False))
    assert count(surface, color, status_bar()) > 0


def test_while_checking_the_status_bar_has_no_verdict() -> None:
    surface = render(square(), Overlays(checking=True))
    for color in (ERROR, WARNING, READY):
        assert count(surface, color, status_bar()) == 0


def test_a_message_shows_above_the_status_bar() -> None:
    surface = render(square(), Overlays(message=Message("Saved tracks/square.json.", READY)))
    above = pygame.Rect(0, SIZE[1] - STATUS_HEIGHT - 50, SIZE[0], 40)
    assert count(surface, READY, above) > 0


def test_a_dialog_shows_in_the_middle_with_what_was_typed() -> None:
    editor = EditorController(TrackDraft(), CAMERA)
    asking = render(editor, Overlays(dialog=DialogBox("Save as:", "Enter: save", "a.json")))
    question = render(editor, Overlays(dialog=DialogBox("Close?", "Q: close")))
    middle = pygame.Rect(SIZE[0] // 2 - 200, SIZE[1] // 2 - 60, 400, 120)
    assert count(asking, SELECTED, middle) > count(question, SELECTED, middle) > 0
