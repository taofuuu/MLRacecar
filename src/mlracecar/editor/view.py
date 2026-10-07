"""Draws the editor: grid, road, points, track problems, a status bar, and panels on top.

The view only reads state (the controller's, plus the window's `Overlays`) and draws it onto any
pygame surface: a window, or an image in memory, which is how the tests check it.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pygame

from mlracecar.core.geometry import FloatArray, IntArray, project_onto_polyline
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import (
    ControlPointAt,
    Location,
    Severity,
    Spot,
    Stretch,
    ValidationIssue,
)
from mlracecar.editor.controller import EditorController, Rounding
from mlracecar.render.drawing import BACKGROUND, Color, draw_grid, draw_track

POINT: Color = (96, 165, 250)
START_POINT: Color = (74, 222, 128)
HOVERED: Color = (191, 219, 254)
SELECTED: Color = (250, 204, 21)
OUTLINE: Color = (100, 116, 139)
TEXT: Color = (226, 232, 240)
MUTED_TEXT: Color = (148, 163, 184)
PANEL: tuple[int, int, int, int] = (15, 23, 42, 225)
ERROR: Color = (239, 68, 68)
WARNING: Color = (251, 146, 60)
READY: Color = START_POINT

SEVERITY_COLORS = {Severity.ERROR: ERROR, Severity.WARNING: WARNING}

POINT_RADIUS = 5
STATUS_HEIGHT = 26
FONT_SIZE = 20
ISSUE_LIST_WIDTH = 460
MAX_LISTED_ISSUES = 6
TOOLTIP_WIDTH = 360
SPOT_RADIUS = 14
"""Pixels: the ring round a problem spot, and round a stretch too short to see when zoomed out."""


class Message(NamedTuple):
    """A short note shown above the status bar for a few seconds, e.g. after saving."""

    text: str
    color: Color = TEXT


class DialogBox(NamedTuple):
    """A question in the middle of the window."""

    title: str
    hint: str
    """Which keys answer it, e.g. "Enter: save   Esc: cancel"."""
    text: str | None = None
    """What's been typed, for questions answered with text; ``None`` for key-only questions."""


@dataclass(frozen=True)
class Overlays:
    """What the window shows besides the editor itself."""

    issues: Sequence[ValidationIssue] = ()
    """Track problems to highlight; possibly found in a slightly older draft."""
    checking: bool = False
    """Whether the track is still being checked and there's nothing to show yet."""
    show_issue_list: bool = True
    help_lines: Sequence[tuple[str, str]] = ()
    """(keys, what they do) for the help panel; empty hides it."""
    message: Message | None = None
    dialog: DialogBox | None = None


NO_OVERLAYS = Overlays()


class EditorView:
    """Draws an `EditorController`'s state. Needs ``pygame.font`` to be initialised."""

    def __init__(self) -> None:
        self._font = pygame.font.Font(None, FONT_SIZE)

    def draw(
        self, surface: pygame.Surface, editor: EditorController, overlays: Overlays = NO_OVERLAYS
    ) -> None:
        """Draw the whole editor, with ``overlays`` on top."""
        surface.fill(BACKGROUND)
        draw_grid(surface, editor.camera, editor.grid_step)
        track = editor.draft.track
        if track is None:
            self._draw_outline(surface, editor)
        else:
            draw_track(surface, track, editor.camera)
        self._draw_issue_locations(surface, editor, overlays.issues)
        self._draw_points(surface, editor)
        self._draw_insert_hint(surface, editor)
        if overlays.show_issue_list and overlays.issues:
            self._draw_issue_list(surface, overlays.issues)
        if overlays.dialog is None:
            self._draw_tooltip(surface, editor, issues_under_cursor(editor, overlays.issues))
        self._draw_status(surface, editor, overlays)
        if overlays.message is not None:
            self._draw_message(surface, overlays.message)
        if overlays.help_lines:
            self._draw_help(surface, overlays.help_lines)
        if overlays.dialog is not None:
            self._draw_dialog(surface, overlays.dialog)

    def _draw_outline(self, surface: pygame.Surface, editor: EditorController) -> None:
        """Until the points make a track, join them with thin lines to show their order."""
        points = _points_on_screen(editor)
        if len(points) >= 2:
            pygame.draw.lines(surface, OUTLINE, len(points) >= 3, points)

    def _draw_issue_locations(
        self, surface: pygame.Surface, editor: EditorController, issues: Sequence[ValidationIssue]
    ) -> None:
        """Mark where each problem is: a ring round a point, coloured road edges along a
        stretch, or a circle on a spot. Errors are drawn last, so they show over warnings."""
        camera, track = editor.camera, editor.draft.track
        for issue in sorted(issues, key=lambda issue: issue.severity is Severity.ERROR):
            color = SEVERITY_COLORS[issue.severity]
            match issue.location:
                case ControlPointAt(point_index=index) if index < len(editor.draft.points):
                    pixel = camera.to_screen(editor.draft.points[index])
                    pygame.draw.circle(surface, color, pixel, POINT_RADIUS + 7, width=3)
                case Stretch(start=start, end=end) if track is not None:
                    samples = _stretch_samples(track, start, end)
                    for edge in (track.left, track.right):
                        pygame.draw.lines(surface, color, False, camera.to_screen(edge[samples]), 4)
                    if (end - start) % track.length * camera.scale < 2 * SPOT_RADIUS:
                        middle = camera.to_screen(
                            track.centerline.points[samples[len(samples) // 2]]
                        )
                        pygame.draw.circle(surface, color, middle, SPOT_RADIUS, width=3)
                case Spot(x=x, y=y):
                    pygame.draw.circle(
                        surface, color, camera.to_screen((x, y)), SPOT_RADIUS, width=3
                    )

    def _draw_points(self, surface: pygame.Surface, editor: EditorController) -> None:
        hovered = editor.hovered
        for index, pixel in enumerate(_points_on_screen(editor)):
            if index == editor.selected:
                pygame.draw.circle(surface, SELECTED, pixel, POINT_RADIUS + 4, width=2)
            color = START_POINT if index == 0 else POINT
            radius = POINT_RADIUS + 1 if index == hovered else POINT_RADIUS
            pygame.draw.circle(surface, HOVERED if index == hovered else color, pixel, radius)

    def _draw_insert_hint(self, surface: pygame.Surface, editor: EditorController) -> None:
        """A hollow dot at the cursor while a click there would insert a point into the road."""
        if editor.hovered is None and editor.draft.is_on_road(editor.cursor_world):
            pygame.draw.circle(surface, HOVERED, editor.cursor, POINT_RADIUS, width=1)

    def _draw_issue_list(self, surface: pygame.Surface, issues: Sequence[ValidationIssue]) -> None:
        """A compact panel in the top-left corner: one line per problem, errors first. The full
        message shows when the cursor is over the problem's marker."""
        width = min(ISSUE_LIST_WIDTH, surface.get_width() - 20)
        errors_first = sorted(issues, key=lambda issue: issue.severity is not Severity.ERROR)
        rows: list[tuple[str, Color, Color | None]] = []  # text, its colour, a bullet's colour
        for issue in errors_first[:MAX_LISTED_ISSUES]:
            line = self._shorten(issue.message, width - 44)
            rows.append((line, TEXT, SEVERITY_COLORS[issue.severity]))
        if len(issues) > MAX_LISTED_ISSUES:
            rows.append((f"... and {len(issues) - MAX_LISTED_ISSUES} more", MUTED_TEXT, None))
        rows.append(("Point at a marker for details.   I: hide this list", MUTED_TEXT, None))

        line_height = self._font.get_linesize() + 2
        panel = pygame.Rect(10, 10, width, line_height * len(rows) + 20)
        self._backdrop(surface, panel)
        for row, (line, color, bullet) in enumerate(rows):
            y = panel.top + 10 + row * line_height
            if bullet is not None:
                pygame.draw.circle(surface, bullet, (panel.left + 18, y + line_height // 2 - 1), 5)
            self._text(surface, line, (panel.left + 32, y), color, anchor="topleft")

    def _draw_tooltip(
        self, surface: pygame.Surface, editor: EditorController, issues: Sequence[ValidationIssue]
    ) -> None:
        """The full messages of the problems under the cursor, next to it."""
        rows = [
            (line, SEVERITY_COLORS[issue.severity] if number == 0 else None)
            for issue in issues
            for number, line in enumerate(self._wrap(issue.message, TOOLTIP_WIDTH - 44))
        ]
        if not rows:
            return
        line_height = self._font.get_linesize() + 2
        box = pygame.Rect(0, 0, TOOLTIP_WIDTH, line_height * len(rows) + 16)
        box.topleft = (round(editor.cursor[0]) + 18, round(editor.cursor[1]) + 18)
        box.clamp_ip(surface.get_rect().inflate(-8, -2 * STATUS_HEIGHT))
        self._backdrop(surface, box)
        for row, (line, bullet) in enumerate(rows):
            y = box.top + 8 + row * line_height
            if bullet is not None:
                pygame.draw.circle(surface, bullet, (box.left + 16, y + line_height // 2 - 1), 5)
            self._text(surface, line, (box.left + 30, y), TEXT, anchor="topleft")

    def _draw_status(
        self, surface: pygame.Surface, editor: EditorController, overlays: Overlays
    ) -> None:
        width, height = surface.get_size()
        bar = pygame.Rect(0, height - STATUS_HEIGHT, width, STATUS_HEIGHT)
        surface.fill(PANEL[:3], bar)

        if editor.rounding is not None:
            text = rounding_status(editor.rounding)
            self._text(surface, text, (10, bar.centery), SELECTED, anchor="midleft")
            return
        draft = editor.draft
        track = draft.track
        x, y = editor.cursor_world
        parts = [
            f"{len(draft.points)} points",
            "not a track yet" if track is None else f"{track.length:.0f} m",
            f"grid {editor.grid_step:g} m" + (", snap on" if editor.snap else ""),
            f"({x:.1f}, {y:.1f})",
        ]
        if editor.selected is not None:
            parts.append(f"point {editor.selected}: {draft.widths[editor.selected]:g} m wide")
        self._text(surface, "   ".join(parts), (10, bar.centery), TEXT, anchor="midleft")
        help_hint = self._text(
            surface, "H: help", (width - 10, bar.centery), MUTED_TEXT, anchor="midright"
        )
        summary, color = _summary(overlays)
        self._text(surface, summary, (help_hint.left - 24, bar.centery), color, anchor="midright")

    def _draw_message(self, surface: pygame.Surface, message: Message) -> None:
        width, height = surface.get_size()
        text_width, text_height = self._font.size(message.text)
        box = pygame.Rect(0, 0, text_width + 32, text_height + 16)
        box.midbottom = (width // 2, height - STATUS_HEIGHT - 12)
        self._backdrop(surface, box)
        self._text(surface, message.text, box.center, message.color, anchor="center")

    def _draw_help(self, surface: pygame.Surface, help_lines: Sequence[tuple[str, str]]) -> None:
        line_height = self._font.get_linesize() + 4
        keys_width = max(self._font.size(keys)[0] for keys, _ in help_lines) + 24
        text_width = max(self._font.size(text)[0] for _, text in help_lines)
        panel = pygame.Rect(0, 0, keys_width + text_width + 40, line_height * len(help_lines) + 32)
        panel.center = surface.get_rect().center
        self._backdrop(surface, panel)
        for row, (keys, text) in enumerate(help_lines):
            y = panel.top + 16 + row * line_height
            self._text(surface, keys, (panel.left + 20, y), SELECTED, anchor="topleft")
            self._text(surface, text, (panel.left + 20 + keys_width, y), TEXT, anchor="topleft")

    def _draw_dialog(self, surface: pygame.Surface, dialog: DialogBox) -> None:
        lines = [(dialog.title, TEXT)]
        if dialog.text is not None:
            lines.append((dialog.text + "|", SELECTED))  # the bar is the typing cursor
        lines.append((dialog.hint, MUTED_TEXT))
        line_height = self._font.get_linesize() + 10
        widest = max(self._font.size(line)[0] for line, _ in lines)
        box = pygame.Rect(0, 0, max(420, widest + 48), line_height * len(lines) + 30)
        box.center = surface.get_rect().center
        self._backdrop(surface, box)
        pygame.draw.rect(surface, SELECTED, box, width=1)
        for row, (line, color) in enumerate(lines):
            at = (box.left + 24, box.top + 18 + row * line_height)
            self._text(surface, line, at, color, anchor="topleft")

    def _backdrop(self, surface: pygame.Surface, area: pygame.Rect) -> None:
        """A see-through dark panel, so the track stays faintly visible behind text."""
        backdrop = pygame.Surface(area.size, pygame.SRCALPHA)
        backdrop.fill(PANEL)
        surface.blit(backdrop, area)

    def _shorten(self, text: str, width: int) -> str:
        """``text`` cut at a word to fit in ``width`` pixels, with "..." if anything was cut."""
        lines = self._wrap(text, width - self._font.size(" ...")[0])
        return lines[0] if len(lines) == 1 else lines[0] + " ..."

    def _wrap(self, text: str, width: int) -> list[str]:
        """Split ``text`` into lines no wider than ``width`` pixels, between words."""
        lines = [""]
        for word in text.split():
            candidate = f"{lines[-1]} {word}".strip()
            if lines[-1] and self._font.size(candidate)[0] > width:
                lines.append(word)
            else:
                lines[-1] = candidate
        return lines

    def _text(
        self,
        surface: pygame.Surface,
        text: str,
        at: tuple[float, float],
        color: Color,
        *,
        anchor: str,
    ) -> pygame.Rect:
        image = self._font.render(text, True, color)
        rect = image.get_rect(**{anchor: at})
        surface.blit(image, rect)
        return rect


def _points_on_screen(editor: EditorController) -> FloatArray:
    return editor.camera.to_screen(np.reshape(editor.draft.points, (-1, 2)))


def _stretch_samples(track: Track, start: float, end: float) -> IntArray:
    """Centerline samples from ``start`` to ``end`` metres along the lap, in driving order
    (through the start/finish line if ``end < start``); at least two, to draw a line."""
    ahead = (track.centerline.arc_length - start) % track.length
    inside = np.flatnonzero(ahead <= (end - start) % track.length)
    if len(inside) >= 2:
        return inside[np.argsort(ahead[inside])]
    first = int(np.argmin(ahead))
    return np.array([first, (first + 1) % len(ahead)])


def rounding_status(rounding: Rounding) -> str:
    """What the status bar says while a corner is being rounded."""
    limits = rounding.limits
    text = f"Corner radius {rounding.radius:.0f} m"
    if rounding.radius >= limits.largest:
        text += " (the widest that fits)"
    elif rounding.radius <= limits.smallest:
        text += " (the tightest the track checks accept)"
    else:
        text += f" (fits {limits.smallest:.0f}-{limits.largest:.0f} m)"
    return text + "   wheel: change   let go of C: keep   Esc: put it back"


def issues_under_cursor(
    editor: EditorController, issues: Sequence[ValidationIssue]
) -> list[ValidationIssue]:
    """The problems whose marker is under the cursor (within about a marker's size)."""
    reach = SPOT_RADIUS / editor.camera.scale
    along = None  # how far along the lap the cursor is, if it's on the road
    track = editor.draft.track
    if track is not None and any(isinstance(issue.location, Stretch) for issue in issues):
        nearest = project_onto_polyline([editor.cursor_world], track.centerline.points, closed=True)
        if abs(nearest.offset[0]) <= track.width[int(nearest.segment[0])] / 2 + reach:
            along = float(nearest.arc_length[0])
    return [issue for issue in issues if _is_under(issue.location, editor, reach, along)]


def _is_under(
    location: Location, editor: EditorController, reach: float, along: float | None
) -> bool:
    """Whether a problem's marker is within ``reach`` metres of the cursor."""
    cursor, points, track = editor.cursor_world, editor.draft.points, editor.draft.track
    match location:
        case ControlPointAt(point_index=index) if index < len(points):
            return math.dist(points[index], cursor) <= reach
        case Spot(x=x, y=y):
            return math.dist((x, y), cursor) <= reach
        case Stretch(start=start, end=end) if track is not None and along is not None:
            # Pad both ends by `reach`, so the ring round a short stretch counts as part of it.
            lap = track.length
            return (along - start + reach) % lap <= (end - start) % lap + 2 * reach
    return False


def _summary(overlays: Overlays) -> tuple[str, Color]:
    """The status bar's verdict on the track."""
    if overlays.checking:
        return "checking...", MUTED_TEXT
    errors = sum(issue.severity is Severity.ERROR for issue in overlays.issues)
    warnings = len(overlays.issues) - errors
    if errors:
        return f"{_count(errors, 'error')}: can't race yet", ERROR
    if warnings:
        return f"{_count(warnings, 'warning')}: ready to race", WARNING
    return "ready to race", READY


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"
