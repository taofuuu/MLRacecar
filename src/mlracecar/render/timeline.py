"""The timeline: a bar along the bottom of a replay showing where it's up to, to click or drag.

Where the bar sits and which point of it a click means are plain functions (`bar_rect`,
`fraction_at`), so they can be tested without drawing; `Timeline` only draws.
"""

import pygame

from mlracecar.render.hud import DIM, FONT_SIZE, MARGIN, PANEL, TEXT, Color

BAR_HEIGHT = 8
"""The bar's height, in pixels."""

GRAB = 10
"""How far above or below the bar a click still grabs it, in pixels."""

CAPTION_SPACE = 2 * MARGIN + FONT_SIZE
"""Room left under the bar for the HUD's caption line."""

PLAYED: Color = (96, 165, 250)
"""The part of the bar already played."""


def bar_rect(size: tuple[int, int]) -> pygame.Rect:
    """Where the bar is in a window of ``size``: across the bottom, above the caption line."""
    width, height = size
    return pygame.Rect(MARGIN, height - CAPTION_SPACE - BAR_HEIGHT, width - 2 * MARGIN, BAR_HEIGHT)


def grabs(size: tuple[int, int], pixel: tuple[int, int]) -> bool:
    """Whether a click at ``pixel`` is on the bar (or close enough to it)."""
    return bar_rect(size).inflate(0, 2 * GRAB).collidepoint(pixel)


def fraction_at(size: tuple[int, int], x: float) -> float:
    """How far along the bar ``x`` is, from 0 (the start) to 1 (the end)."""
    bar = bar_rect(size)
    return min(max((x - bar.left) / max(bar.width, 1), 0.0), 1.0)


class Timeline:
    """Draws the timeline."""

    def __init__(self) -> None:
        pygame.font.init()  # works without a window
        self._font = pygame.font.Font(None, FONT_SIZE)

    def draw(self, surface: pygame.Surface, fraction: float, label: str) -> None:
        """Draw the bar with ``fraction`` of it played, and ``label`` above its right end."""
        bar = bar_rect(surface.get_size())
        backdrop = pygame.Surface((bar.width, bar.height), pygame.SRCALPHA)
        backdrop.fill(PANEL)
        surface.blit(backdrop, bar.topleft)
        played = bar.copy()
        played.width = round(bar.width * min(max(fraction, 0.0), 1.0))
        pygame.draw.rect(surface, PLAYED, played)
        pygame.draw.circle(surface, TEXT, (played.right, bar.centery), BAR_HEIGHT)
        image = self._font.render(label, True, DIM)
        surface.blit(image, (bar.right - image.get_width(), bar.top - image.get_height() - 6))
