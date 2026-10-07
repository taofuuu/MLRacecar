"""The heads-up display: one car's speed, lap, and lap times, drawn over the race.

What the HUD says is worked out by plain functions (`hud_lines`, `warnings`), so it can be
tested without drawing anything; `Hud` only draws it.
"""

import math

import pygame

from mlracecar.core.snapshot import Snapshot

type Color = tuple[int, int, int]

TEXT: Color = (235, 238, 242)
DIM: Color = (150, 156, 164)
INVALID: Color = (248, 113, 113)
OFF_TRACK: Color = (251, 146, 60)
WRONG_WAY: Color = (239, 68, 68)
PANEL: tuple[int, int, int, int] = (12, 14, 18, 190)

FONT_SIZE = 24
BIG_FONT_SIZE = 44
MARGIN = 12


def lap_time_text(seconds: float) -> str:
    """A lap time as ``m:ss.mmm``; dashes when there's no time yet (NaN)."""
    if math.isnan(seconds):
        return "-:--.---"
    milliseconds = round(seconds * 1000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    return f"{minutes}:{milliseconds // 1000:02d}.{milliseconds % 1000:03d}"


def hud_lines(snapshot: Snapshot, car: int) -> list[tuple[str, Color]]:
    """What the HUD shows for one car, line by line, with each line's colour."""
    race = snapshot.race
    speed = f"{snapshot.cars.speed[car] * 3.6:.0f} km/h"
    if race.checkpoint[car] < 0:
        lap, current, current_color = "Out lap", "-:--.---", DIM
    else:
        lap = f"Lap {race.laps[car] + 1}"
        current = lap_time_text(snapshot.time - float(race.lap_start[car]))
        current_color = TEXT if race.clean[car] else INVALID
        if not race.clean[car]:
            current += "  invalid"
    return [
        (speed, TEXT),
        (lap, TEXT),
        (f"Time  {current}", current_color),
        (f"Last  {lap_time_text(float(race.last_lap[car]))}", DIM),
        (f"Best  {lap_time_text(float(race.best_lap[car]))}", DIM),
    ]


def warnings(snapshot: Snapshot, car: int) -> list[tuple[str, Color]]:
    """Big warnings for one car: out of the race, off the track, the wrong way."""
    race = snapshot.race
    shown = []
    if race.out[car]:
        shown.append(("OUT", WRONG_WAY))
    elif race.off_track[car]:
        shown.append(("OFF TRACK", OFF_TRACK))
    if race.wrong_way[car]:
        shown.append(("WRONG WAY", WRONG_WAY))
    return shown


class Hud:
    """Draws the HUD."""

    def __init__(self) -> None:
        pygame.font.init()  # works without a window
        self._font = pygame.font.Font(None, FONT_SIZE)
        self._big = pygame.font.Font(None, BIG_FONT_SIZE)

    def draw(self, surface: pygame.Surface, snapshot: Snapshot, car: int, caption: str) -> None:
        """Draw one car's HUD, with ``caption`` (the camera, say) along the bottom."""
        lines = hud_lines(snapshot, car)
        images = [self._big.render(lines[0][0], True, lines[0][1])]
        images += [self._font.render(text, True, color) for text, color in lines[1:]]
        width = max(image.get_width() for image in images) + 2 * MARGIN
        height = sum(image.get_height() + 4 for image in images) + 2 * MARGIN
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(PANEL)
        surface.blit(panel, (MARGIN, MARGIN))
        y = 2 * MARGIN
        for image in images:
            surface.blit(image, (2 * MARGIN, y))
            y += image.get_height() + 4

        y = MARGIN
        for text, color in warnings(snapshot, car):
            image = self._big.render(text, True, color)
            surface.blit(image, ((surface.get_width() - image.get_width()) // 2, y))
            y += image.get_height()

        image = self._font.render(caption, True, DIM)
        surface.blit(image, (MARGIN, surface.get_height() - image.get_height() - MARGIN))
