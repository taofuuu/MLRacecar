"""Tests for mlracecar.render.timeline: where the bar is and what a click on it means."""

import pygame
import pytest

from mlracecar.render.timeline import Timeline, bar_rect, fraction_at, grabs

SIZE = (800, 600)


def test_the_bar_runs_across_the_bottom_above_the_caption() -> None:
    bar = bar_rect(SIZE)

    assert (bar.left, bar.right) == (12, 788)
    assert 500 < bar.top < bar.bottom < 600 - 30


@pytest.mark.parametrize(
    ("x", "fraction"), [(12, 0.0), (400, 0.5), (788, 1.0), (0, 0.0), (900, 1.0)]
)
def test_a_point_along_the_bar_is_a_fraction_of_the_replay(x: int, fraction: float) -> None:
    assert fraction_at(SIZE, x) == pytest.approx(fraction)


def test_a_click_near_the_bar_grabs_it() -> None:
    bar = bar_rect(SIZE)

    assert grabs(SIZE, (400, bar.centery))
    assert grabs(SIZE, (400, bar.top - 8))
    assert not grabs(SIZE, (400, bar.top - 30))
    assert not grabs(SIZE, (400, 100))


def test_the_bar_shows_how_much_has_played() -> None:
    surface = pygame.Surface(SIZE)
    bar = bar_rect(SIZE)

    Timeline().draw(surface, 0.5, "0:30.000 / 1:00.000  x1")

    assert surface.get_at((bar.left + 100, bar.centery))[:3] == (96, 165, 250)
    assert surface.get_at((bar.right - 100, bar.centery))[:3] != (96, 165, 250)
