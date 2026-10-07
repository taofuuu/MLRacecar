"""How fast the distance sensors are: reading every car's 15 rays, for 1, 64, and 1,024 cars.

Run with `uv run pytest -m benchmark --no-cov`; `scripts/benchmark_table.py` turns the results
into the table in the README.

The cars are where the scripted driver has taken them after ten seconds of racing on the GP
circuit, from the start grid, so the rays meet the edges the way they do in a race.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from drivers import CenterlineDriver
from mlracecar.config.models import SensorConfig, SimulationConfig, VehicleConfig
from mlracecar.core.sensors import RaySensor
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.io.track_file import read_track_file

TRACK = read_track_file(Path(__file__).parents[2] / "tracks" / "gp-circuit.json").to_track()
CAR = VehicleConfig().to_params()
TIMING = SimulationConfig().to_timing()
SETTINGS = SensorConfig().to_settings()


@cache
def racing(cars: int) -> Snapshot:
    """``cars`` cars ten seconds into a race."""
    world = World(TRACK, KinematicBicycle(CAR), TIMING, cars, np.random.default_rng(0))
    driver = CenterlineDriver(TRACK, CAR, 60.0)
    for _ in range(200):
        world.step(driver.act(world.snapshot))
    return world.snapshot


@pytest.mark.parametrize("cars", [1, 64, 1024])
def test_sense(benchmark: BenchmarkFixture, cars: int) -> None:
    sensor = RaySensor(TRACK, SETTINGS)
    snapshot = racing(cars)

    readings = benchmark(sensor.sense, snapshot)
    benchmark.extra_info.update(cars=cars, rays=SETTINGS.count)

    assert readings.distance.shape == (cars, SETTINGS.count)
    assert benchmark.stats is not None
    assert benchmark.stats.stats.mean < TIMING.decision_dt  # faster than real time
