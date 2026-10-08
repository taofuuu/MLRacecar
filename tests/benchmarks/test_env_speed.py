"""How fast the RL environment runs: one step of every car, with observations, rewards, and runs
ending and starting again, for one world of 1, 64, and 1,024 cars, and for 64 separate
single-car environments (Gymnasium's `SyncVectorEnv`).

Run with `uv run pytest -m benchmark --no-cov`; `scripts/benchmark_table.py` turns the results
into the table in the README.

The cars start at random places on the technical track and drive with fixed random actions,
mostly forwards, so runs keep ending (off the road, stuck, out of time) and starting again, as
they do in training.
"""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.vector import SyncVectorEnv, VectorEnv
from pytest_benchmark.fixture import BenchmarkFixture

from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.play.environment import make_racing_env

TECHNICAL = Path(__file__).parents[2] / "tracks" / "technical.json"


def driving(env: VectorEnv, cars: int) -> np.ndarray:
    """Reset ``env`` with random starts, drive it for a second, and return the actions it's
    being driven with."""
    env.reset(seed=0, options={"start": "random"})
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (cars, 2))
    actions[:, 1] = 0.3 + 0.7 * np.abs(actions[:, 1])
    for _ in range(20):
        env.step(actions)
    return actions


@pytest.mark.parametrize("cars", [1, 64, 1024])
def test_one_world(benchmark: BenchmarkFixture, cars: int) -> None:
    env = BatchedRacingEnv(TECHNICAL, cars)
    actions = driving(env, cars)

    benchmark(env.step, actions)
    benchmark.extra_info.update(cars=cars, kind="batched")


@pytest.mark.parametrize("cars", [64])
def test_separate_environments(benchmark: BenchmarkFixture, cars: int) -> None:
    env = SyncVectorEnv([lambda: make_racing_env(TECHNICAL) for _ in range(cars)])
    actions = driving(env, cars)

    benchmark(env.step, actions)
    benchmark.extra_info.update(cars=cars, kind="separate")
