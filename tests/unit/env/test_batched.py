"""Tests for mlracecar.env.batched: many cars in one world, exactly like separate environments."""

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import pytest
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from mlracecar.config.models import EpisodeConfig, RacecarConfig
from mlracecar.env import ENV_ID
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.play.environment import make_racing_env

TECHNICAL = Path(__file__).parents[3] / "tracks" / "technical.json"
OVAL = TECHNICAL.parent / "oval.json"
CARS = 8
SHORT_RUNS = RacecarConfig(episode=EpisodeConfig(time_limit=4.0, stuck_time=1.0))
"""Runs short enough to end, in every way, many times in a few hundred steps."""


def mostly_forward(rng: np.random.Generator, cars: int = CARS) -> np.ndarray:
    """Random actions with the pedal never backwards, so cars get somewhere and leave the road."""
    actions = rng.uniform(-1.0, 1.0, (cars, 2))
    actions[:, 1] = np.abs(actions[:, 1])
    return actions


def batched(mode: AutoresetMode = AutoresetMode.NEXT_STEP, **kwargs: Any) -> BatchedRacingEnv:
    return BatchedRacingEnv(
        TECHNICAL, CARS, kwargs.pop("config", SHORT_RUNS), autoreset_mode=mode, **kwargs
    )


# --------------------------------------------------------------------------- #
# The standard
# --------------------------------------------------------------------------- #


def test_gymnasium_makes_it_for_many_cars() -> None:
    env = gymnasium.make_vec(
        ENV_ID,
        num_envs=4,
        vectorization_mode="vector_entry_point",
        track=TECHNICAL,
        autoreset_mode="SameStep",
    )

    assert isinstance(env.unwrapped, BatchedRacingEnv)
    assert env.num_envs == 4
    assert env.unwrapped.autoreset_mode is AutoresetMode.SAME_STEP


def test_spaces_hold_one_row_per_car() -> None:
    env = batched()

    assert env.single_action_space.shape == (2,)
    assert env.action_space.shape == (CARS, 2)
    assert env.single_observation_space.shape == (31,)
    assert env.observation_space.shape == (CARS, 31)
    assert env.metadata["autoreset_mode"] is AutoresetMode.NEXT_STEP


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_envs": 0}, "at least 1 car"),
        ({"num_envs": 2, "autoreset_mode": AutoresetMode.DISABLED}, "NEXT_STEP or SAME_STEP"),
    ],
)
def test_impossible_setups_are_refused(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BatchedRacingEnv(TECHNICAL, **kwargs)


def test_stepping_before_the_first_reset_is_refused() -> None:
    with pytest.raises(gymnasium.error.ResetNeeded):
        batched().step(np.zeros((CARS, 2)))


def test_actions_need_a_row_per_car() -> None:
    env = batched()
    env.reset(seed=0)

    with pytest.raises(ValueError, match="actions"):
        env.step(np.zeros((CARS - 1, 2)))


def test_seeds_must_be_one_per_car() -> None:
    with pytest.raises(ValueError, match=f"expected {CARS} seeds"):
        batched().reset(seed=[1, 2, 3])


# --------------------------------------------------------------------------- #
# Exactly like separate environments
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", [AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP])
@pytest.mark.parametrize("start", ["grid", "random"])
def test_every_car_runs_exactly_as_it_would_alone(mode: AutoresetMode, start: str) -> None:
    together = batched(mode)
    alone = SyncVectorEnv(
        [lambda: make_racing_env(TECHNICAL, SHORT_RUNS) for _ in range(CARS)], autoreset_mode=mode
    )
    observations, _ = together.reset(seed=5, options={"start": start})
    expected, _ = alone.reset(seed=5, options={"start": start})
    np.testing.assert_array_equal(observations, expected)
    rng = np.random.default_rng(0)
    ends = 0

    for _ in range(150):
        actions = mostly_forward(rng)
        observations, rewards, terminated, truncated, info = together.step(actions)
        *theirs, their_info = alone.step(actions)
        for mine, other in zip((observations, rewards, terminated, truncated), theirs, strict=True):
            np.testing.assert_array_equal(mine, other)
        ended = terminated | truncated
        ends += int(ended.sum())
        for car in np.flatnonzero(ended):
            if mode is AutoresetMode.SAME_STEP:
                np.testing.assert_array_equal(info["final_obs"][car], their_info["final_obs"][car])
                assert (
                    info["final_info"]["end_reason"][car]
                    == their_info["final_info"]["end_reason"][car]
                )
            else:
                assert info["end_reason"][car] == their_info["end_reason"][car]
        np.testing.assert_array_equal(info["distance"], their_info["distance"])

    assert ends >= 10  # runs ended and restarted plenty of times along the way


def test_seeds_given_per_car_work_like_separate_seeds() -> None:
    together = batched()
    alone = SyncVectorEnv([lambda: make_racing_env(TECHNICAL, SHORT_RUNS) for _ in range(CARS)])
    seeds: list[int | None] = list(range(100, 100 + CARS))

    observations, _ = together.reset(seed=seeds, options={"start": "random"})
    expected, _ = alone.reset(seed=seeds, options={"start": "random"})
    np.testing.assert_array_equal(observations, expected)

    again, _ = together.reset(
        options={"start": "random"}
    )  # each car carries on with its own generator
    expected_again, _ = alone.reset(options={"start": "random"})
    np.testing.assert_array_equal(again, expected_again)
    assert not np.array_equal(again, observations)


# --------------------------------------------------------------------------- #
# Starting again
# --------------------------------------------------------------------------- #


def drive_until_some_end(env: BatchedRacingEnv) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Step with mostly-forward actions until some runs end; that step's actions and results."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        actions = mostly_forward(rng)
        result = env.step(actions)
        if (result[2] | result[3]).any():
            return actions, result
    raise AssertionError("no run ended")


def test_next_step_a_car_whose_run_ended_restarts_on_its_next_step() -> None:
    env = batched(AutoresetMode.NEXT_STEP)
    env.reset(seed=0)
    actions, (_, _, terminated, truncated, info) = drive_until_some_end(env)
    ended = terminated | truncated
    assert (info["distance"][ended] > 0).all()  # the step that ends a run still shows it
    assert info["_end_reason"].tolist() == ended.tolist()

    observations, rewards, terminated, truncated, info = env.step(actions)

    assert (rewards[ended] == 0.0).all()
    assert not (terminated | truncated)[ended].any()
    assert (info["distance"][ended] == 0.0).all()
    assert not info["terms"]["_progress"][ended].any()
    assert env.observation_space.contains(observations)


def test_same_step_a_car_whose_run_ended_restarts_straight_away() -> None:
    env = batched(AutoresetMode.SAME_STEP)
    env.reset(seed=0)

    _, (observations, _, terminated, truncated, info) = drive_until_some_end(env)

    ended = terminated | truncated
    assert info["_final_obs"].tolist() == ended.tolist()
    assert info["_final_info"].tolist() == ended.tolist()
    final = info["final_info"]
    assert (final["distance"][ended] > 0).all()  # the run that ended
    assert set(final["end_reason"][ended]) <= {"off_track", "time_limit", "stuck"}
    assert (info["distance"][ended] == 0.0).all()  # the run that started
    for car in np.flatnonzero(ended):
        assert not np.array_equal(info["final_obs"][car], observations[car])
    assert env.observation_space.contains(observations)


def test_a_run_reports_its_points_per_term_when_it_ends() -> None:
    env = batched(AutoresetMode.NEXT_STEP)
    env.reset(seed=0)
    rng = np.random.default_rng(2)
    progress = np.zeros(CARS)

    for _ in range(200):
        _, _, terminated, truncated, info = env.step(mostly_forward(rng))
        progress += np.where(info["terms"]["_progress"], info["terms"]["progress"], 0.0)
        ended = terminated | truncated
        if ended.any():
            break

    car = int(np.flatnonzero(ended)[0])
    assert info["episode_terms"]["progress"][car] == pytest.approx(progress[car])
    assert info["episode_terms"]["_progress"].tolist() == ended.tolist()


def test_a_reset_can_change_the_track_for_every_car() -> None:
    env = batched()

    observations, info = env.reset(seed=0, options={"track": OVAL})

    assert env.track.length != batched().track.length
    assert env.observation_space.contains(observations)
    assert info["_distance"].all()


def test_a_random_agent_drives_64_cars_without_errors() -> None:
    env = BatchedRacingEnv(TECHNICAL, 64, SHORT_RUNS, autoreset_mode=AutoresetMode.SAME_STEP)
    env.action_space.seed(0)
    env.reset(seed=0, options={"start": "random"})

    for _ in range(100):
        observations, rewards, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observations)
        assert np.isfinite(rewards).all()
