"""Tests for mlracecar.render.hud: what the HUD says, and that it gets drawn."""

from dataclasses import replace

import numpy as np
import pygame
import pytest

from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.render.hud import (
    DIM,
    INVALID,
    OFF_TRACK,
    TEXT,
    WRONG_WAY,
    Hud,
    hud_lines,
    lap_time_text,
    warnings,
)

ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)
TRACK = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 12)
CAR = VehicleConfig().to_params()


def snapshot(**race: object) -> Snapshot:
    """A car on the grid at 10 s and 72 km/h, with some of its race's arrays replaced."""
    world = World(
        TRACK, KinematicBicycle(CAR), SimulationConfig().to_timing(), 1, np.random.default_rng(0)
    )
    taken = world.snapshot
    arrays = {name: np.array([value]) for name, value in race.items()}
    moving = replace(taken.cars, vx=np.array([20.0]))
    return replace(taken, time=10.0, cars=moving, race=replace(taken.race, **arrays))


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (83.4567, "1:23.457"),
        (5.0, "0:05.000"),
        (59.9996, "1:00.000"),
        (600.0, "10:00.000"),
        (float("nan"), "-:--.---"),
    ],
)
def test_lap_times_read_as_minutes_seconds_and_milliseconds(seconds: float, text: str) -> None:
    assert lap_time_text(seconds) == text


def test_before_the_first_lap_the_car_is_on_its_out_lap() -> None:
    lines = hud_lines(snapshot(checkpoint=-1, lap_start=np.nan), 0)

    assert lines[1] == ("Out lap", TEXT)
    assert lines[2] == ("Time  -:--.---", DIM)


def test_during_a_lap_the_hud_shows_its_number_and_time() -> None:
    taken = snapshot(checkpoint=3, lap_start=4.0, laps=2, last_lap=81.25, best_lap=80.5)

    assert hud_lines(taken, 0) == [
        ("72 km/h", TEXT),
        ("Lap 3", TEXT),
        ("Time  0:06.000", TEXT),
        ("Last  1:21.250", DIM),
        ("Best  1:20.500", DIM),
    ]


def test_an_invalid_lap_shows_in_red() -> None:
    lines = hud_lines(snapshot(checkpoint=3, lap_start=4.0, clean=False), 0)

    assert lines[2] == ("Time  0:06.000  invalid", INVALID)


@pytest.mark.parametrize(
    ("flags", "shown"),
    [
        ({}, []),
        ({"off_track": True}, [("OFF TRACK", OFF_TRACK)]),
        ({"wrong_way": True}, [("WRONG WAY", WRONG_WAY)]),
        ({"off_track": True, "wrong_way": True},
         [("OFF TRACK", OFF_TRACK), ("WRONG WAY", WRONG_WAY)]),
        ({"off_track": True, "out": True}, [("OUT", WRONG_WAY)]),
    ],
)  # fmt: skip
def test_warnings_say_what_is_wrong(flags: dict[str, bool], shown: list[object]) -> None:
    assert warnings(snapshot(**flags), 0) == shown


def test_the_hud_is_drawn_in_its_corner_with_its_caption() -> None:
    surface = pygame.Surface((640, 400))
    surface.fill((0, 0, 0))

    Hud().draw(surface, snapshot(wrong_way=True), 0, "camera: follow")

    panel = pygame.surfarray.array3d(surface)[12:140, 12:120]
    assert panel.any()  # the panel and its text
    assert pygame.surfarray.array3d(surface)[260:380, 10:60].any()  # the big warning, centred
    assert pygame.surfarray.array3d(surface)[:200, 370:400].any()  # the caption, bottom left
