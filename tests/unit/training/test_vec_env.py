"""Tests for mlracecar.training.vec_env: the many-cars environment as Stable-Baselines3 sees it.

Skipped where the training libraries aren't installed (``uv sync --extra train`` or
``--extra train-cpu``).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from gymnasium.vector import AutoresetMode
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mlracecar.config.models import EpisodeConfig, RacecarConfig
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.play.environment import make_racing_env
from mlracecar.training.vec_env import SB3VecEnv

TECHNICAL = Path(__file__).parents[3] / "tracks" / "technical.json"
CARS = 8
SHORT_RUNS = RacecarConfig(episode=EpisodeConfig(time_limit=4.0, stuck_time=1.0))


def ours(cars: int = CARS, config: RacecarConfig = SHORT_RUNS) -> SB3VecEnv:
    return SB3VecEnv(
        BatchedRacingEnv(TECHNICAL, cars, config, autoreset_mode=AutoresetMode.SAME_STEP)
    )


def test_it_needs_runs_restarted_in_the_same_step() -> None:
    with pytest.raises(ValueError, match=r"autoreset_mode=AutoresetMode\.SAME_STEP"):
        SB3VecEnv(BatchedRacingEnv(TECHNICAL, 2))


@pytest.mark.parametrize("start", ["grid", "random"])
def test_every_transition_matches_stable_baselines3s_own_vector_env(start: str) -> None:
    together = ours()
    separate = DummyVecEnv([lambda: make_racing_env(TECHNICAL, SHORT_RUNS) for _ in range(CARS)])
    for env in (together, separate):
        env.seed(5)
        env.set_options({"start": start})
    np.testing.assert_array_equal(together.reset(), separate.reset())
    rng = np.random.default_rng(0)
    ends = 0

    for _ in range(150):
        actions = rng.uniform(-1.0, 1.0, (CARS, 2)).astype(np.float32)
        actions[:, 1] = np.abs(actions[:, 1])
        for env in (together, separate):
            env.step_async(actions)
        mine, theirs = together.step_wait(), separate.step_wait()
        for one, other in zip(mine[:3], theirs[:3], strict=True):
            np.testing.assert_array_equal(one, other)
        for car, (info, their_info) in enumerate(zip(mine[3], theirs[3], strict=True)):
            assert info["TimeLimit.truncated"] == their_info["TimeLimit.truncated"]
            assert info["distance"] == their_info["distance"]
            if mine[2][car]:
                ends += 1
                np.testing.assert_array_equal(
                    info["terminal_observation"], their_info["terminal_observation"]
                )
                assert info["end_reason"] == their_info["end_reason"]
                assert info["episode_terms"] == pytest.approx(their_info["episode_terms"])
                assert together.reset_infos[car]["distance"] == 0.0

    assert ends >= 10


def test_all_cars_must_share_their_reset_options() -> None:
    env = ours(cars=2)
    env.set_options([{"start": "grid"}, {"start": "random"}])

    with pytest.raises(ValueError, match="give them all the same reset options"):
        env.reset()


def test_attributes_and_methods_are_the_shared_environments() -> None:
    env = ours(cars=3)

    assert env.get_attr("num_envs") == [3, 3, 3]
    assert env.get_attr("num_envs", indices=1) == [3]
    env.set_attr("spare", "value")
    assert env.env.spare == "value"  # type: ignore[attr-defined]
    assert env.env_method("reset", seed=0, indices=[0, 2])[0][0].shape == (3, 31)
    assert env.env_is_wrapped(object) == [False, False, False]
    env.close()


def test_ppo_learns_on_it() -> None:
    env = ours(cars=16, config=RacecarConfig())
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=128, n_epochs=1, device="cpu", seed=0)

    model.learn(total_timesteps=1024)

    assert model.num_timesteps >= 1024
