"""How long drawing a race frame takes: the race window must manage 60 frames a second.

Run with `uv run pytest -m benchmark --no-cov`.
"""

from pathlib import Path

import numpy as np
import pygame
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from drivers import CenterlineDriver
from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.io.track_file import read_track_file
from mlracecar.render.race import CameraMode, Overlay, RaceRenderer

GP_CIRCUIT = Path(__file__).parents[2] / "tracks" / "gp-circuit.json"
CAR = VehicleConfig().to_params()
FRAME = 1 / 60


@pytest.fixture(scope="module")
def race() -> tuple[RaceRenderer, Snapshot]:
    """16 cars 20 s into a race on the GP circuit, and a renderer for a 1280 x 800 window."""
    track = read_track_file(GP_CIRCUIT).to_track()
    world = World(
        track, KinematicBicycle(CAR), SimulationConfig().to_timing(), 16, np.random.default_rng(0)
    )
    driver = CenterlineDriver(track, CAR, 60.0)
    for _ in range(400):
        world.step(driver.act(world.snapshot))
    return RaceRenderer(track, (CAR.length, CAR.width)), world.snapshot


@pytest.mark.parametrize("mode", list(CameraMode))
def test_draw_sixteen_cars(
    benchmark: BenchmarkFixture, race: tuple[RaceRenderer, Snapshot], mode: CameraMode
) -> None:
    renderer, snapshot = race
    renderer.mode = mode
    renderer.overlays = set(Overlay)  # everything on: the slowest case
    surface = pygame.Surface(renderer.size)

    benchmark(renderer.draw, surface, snapshot)

    assert benchmark.stats is not None
    assert benchmark.stats.stats.mean < FRAME
