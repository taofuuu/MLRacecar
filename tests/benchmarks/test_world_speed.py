"""How fast the simulation runs: one world step (a driver decision) for 1, 64, and 1,024 cars.

Run with `uv run pytest -m benchmark --no-cov`; `scripts/benchmark_table.py` turns the results
into the table in the README.

Every car is driven by the scripted driver from a standing start on the GP circuit, so each
measured step is a real step of a race: six physics updates and the race rules, for every car.
The driver decides between steps, outside the timing, and the number of steps is fixed, so every
run measures exactly the same steps.
"""

from pathlib import Path

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from drivers import CenterlineDriver
from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.geometry import FloatArray
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.io.track_file import read_track_file

GP_CIRCUIT = Path(__file__).parents[2] / "tracks" / "gp-circuit.json"
CAR = VehicleConfig().to_params()
TIMING = SimulationConfig().to_timing()
WARMUP_STEPS = 20  # the first second of racing, not measured
MEASURED_STEPS = 200  # the next ten seconds


@pytest.fixture(scope="module")
def track() -> Track:
    return read_track_file(GP_CIRCUIT).to_track()


@pytest.mark.parametrize("cars", [1, 64, 1024])
def test_world_step(benchmark: BenchmarkFixture, track: Track, cars: int) -> None:
    world = World(track, KinematicBicycle(CAR), TIMING, cars, np.random.default_rng(0))
    driver = CenterlineDriver(track, CAR, 60.0)

    def next_decision() -> tuple[tuple[FloatArray], dict[str, object]]:
        return (driver.act(world.snapshot),), {}

    benchmark.pedantic(  # type: ignore[no-untyped-call]  # pytest-benchmark has no hints for it
        world.step, setup=next_decision, rounds=MEASURED_STEPS, warmup_rounds=WARMUP_STEPS
    )
    benchmark.extra_info.update(cars=cars, decision_dt=TIMING.decision_dt)

    assert benchmark.stats is not None
    assert benchmark.stats.stats.mean < TIMING.decision_dt  # faster than real time
