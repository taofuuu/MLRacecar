"""Draws the editor: grid, road, points, a status bar, and the help panel.

The view only reads the controller's state and draws it onto any pygame surface (a window or an
image in memory, which is how the tests check it).
"""

from collections.abc import Sequence

import numpy as np
import pygame

from mlracecar.core.geometry import FloatArray
from mlracecar.editor.controller import EditorController
from mlracecar.render.drawing import BACKGROUND, Color, draw_grid, draw_track

POINT: Color = (96, 165, 250)
START_POINT: Color = (74, 222, 128)
HOVERED: Color = (191, 219, 254)
SELECTED: Color = (250, 204, 21)
OUTLINE: Color = (100, 116, 139)
TEXT: Color = (226, 232, 240)
MUTED_TEXT: Color = (148, 163, 184)
PANEL: tuple[int, int, int, int] = (15, 23, 42, 225)

POINT_RADIUS = 5
STATUS_HEIGHT = 26
FONT_SIZE = 20


class EditorView:
    """Draws an `EditorController`'s state. Needs ``pygame.font`` to be initialised."""

    def __init__(self) -> None:
        self._font = pygame.font.Font(None, FONT_SIZE)

    def draw(
        self,
        surface: pygame.Surface,
        editor: EditorController,
        help_lines: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Draw the whole editor; ``help_lines`` (keys, what they do) shows the help panel."""
        surface.fill(BACKGROUND)
        draw_grid(surface, editor.camera, editor.grid_step)
        track = editor.draft.track
        if track is None:
            self._draw_outline(surface, editor)
        else:
            draw_track(surface, track, editor.camera)
        self._draw_points(surface, editor)
        self._draw_insert_hint(surface, editor)
        self._draw_status(surface, editor)
        if help_lines:
            self._draw_help(surface, help_lines)

    def _draw_outline(self, surface: pygame.Surface, editor: EditorController) -> None:
        """Until the points make a track, join them with thin lines to show their order."""
        points = _points_on_screen(editor)
        if len(points) >= 2:
            pygame.draw.lines(surface, OUTLINE, len(points) >= 3, points)

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

    def _draw_status(self, surface: pygame.Surface, editor: EditorController) -> None:
        width, height = surface.get_size()
        bar = pygame.Rect(0, height - STATUS_HEIGHT, width, STATUS_HEIGHT)
        surface.fill(PANEL[:3], bar)

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
        self._text(surface, "H: help", (width - 10, bar.centery), MUTED_TEXT, anchor="midright")

    def _draw_help(self, surface: pygame.Surface, help_lines: Sequence[tuple[str, str]]) -> None:
        line_height = self._font.get_linesize() + 4
        keys_width = max(self._font.size(keys)[0] for keys, _ in help_lines) + 24
        text_width = max(self._font.size(text)[0] for _, text in help_lines)
        panel = pygame.Rect(0, 0, keys_width + text_width + 40, line_height * len(help_lines) + 32)
        panel.center = surface.get_rect().center
        backdrop = pygame.Surface(panel.size, pygame.SRCALPHA)
        backdrop.fill(PANEL)
        surface.blit(backdrop, panel)
        for row, (keys, text) in enumerate(help_lines):
            y = panel.top + 16 + row * line_height
            self._text(surface, keys, (panel.left + 20, y), SELECTED, anchor="topleft")
            self._text(surface, text, (panel.left + 20 + keys_width, y), TEXT, anchor="topleft")

    def _text(
        self,
        surface: pygame.Surface,
        text: str,
        at: tuple[float, float],
        color: Color,
        *,
        anchor: str,
    ) -> None:
        image = self._font.render(text, True, color)
        rect = image.get_rect(**{anchor: at})
        surface.blit(image, rect)


def _points_on_screen(editor: EditorController) -> FloatArray:
    return editor.camera.to_screen(np.reshape(editor.draft.points, (-1, 2)))
