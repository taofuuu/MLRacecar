"""Tests for mlracecar.training.metrics: the racing numbers a training run tracks."""

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from gymnasium.vector import AutoresetMode

from mlracecar.config.models import REWARD_TERMS, EpisodeConfig, RacecarConfig
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.env.racing import load_track
from mlracecar.training.evaluation import RunResult, summarize
from mlracecar.training.metrics import PracticeRuns, evaluation_scalars, scalars
from mlracecar.training.vec_env import SB3VecEnv

TECHNICAL = load_track(Path(__file__).parents[3] / "tracks" / "technical.json")
DT = 0.05


def ended(laps: int = 0, best_lap: float = math.nan, reason: str = "off_track") -> dict[str, Any]:
    """A car's ``info`` on the step its run ends, as `SB3VecEnv` gives it."""
    terms = dict.fromkeys(REWARD_TERMS, 0.0) | {"progress": 12.0, "off_track": -10.0}
    return {
        "distance": 120.0,
        "laps": laps,
        "best_lap": best_lap,
        "end_reason": reason,
        "episode_terms": terms,
    }


def test_a_practice_run_is_kept_when_it_ends_with_how_long_it_lasted() -> None:
    practice = PracticeRuns(DT)
    driving: dict[str, Any] = {}

    practice.add([driving, driving], np.array([False, False]))
    practice.add([driving, ended()], np.array([False, True]))
    practice.add([ended(1, 30.5, "time_limit"), driving], np.array([True, False]))

    first, second = practice.runs
    assert replace(first, seconds=0.0) == RunResult(
        reward=2.0,
        distance=120.0,
        laps=0,
        best_lap=None,
        end_reason="off_track",
        seconds=0.0,
        terms=ended()["episode_terms"],
    )
    assert first.seconds == pytest.approx(2 * DT)
    assert (second.laps, second.best_lap, second.end_reason) == (1, 30.5, "time_limit")
    assert second.seconds == pytest.approx(3 * DT)


def test_each_car_counts_its_own_run_from_its_last_restart() -> None:
    practice = PracticeRuns(DT)

    practice.add([ended(), {}], np.array([True, False]))
    for _ in range(3):
        practice.add([{}, {}], np.array([False, False]))
    practice.add([ended(), ended()], np.array([True, True]))

    assert [run.seconds for run in practice.runs] == pytest.approx([DT, 4 * DT, 5 * DT])


def test_only_the_latest_runs_count() -> None:
    practice = PracticeRuns(DT, window=3)

    for laps in range(5):
        practice.add([ended(laps)], np.array([True]))

    assert [run.laps for run in practice.runs] == [2, 3, 4]


def test_no_numbers_before_the_first_run_ends() -> None:
    practice = PracticeRuns(DT)
    practice.add([{}], np.array([False]))

    assert practice.scalars() == {}


def test_practice_runs_from_the_environment_add_up_to_its_rewards() -> None:
    config = RacecarConfig(episode=EpisodeConfig(time_limit=2.0))
    env = SB3VecEnv(BatchedRacingEnv(TECHNICAL, 4, config, autoreset_mode=AutoresetMode.SAME_STEP))
    env.seed(5)
    env.reset()
    practice = PracticeRuns(env.env.timing.decision_dt)
    rng = np.random.default_rng(0)
    returns, totals = np.zeros(4), []

    for _ in range(120):
        actions = np.column_stack([rng.uniform(-0.2, 0.2, 4), np.full(4, 0.6)])
        _, rewards, dones, infos = env.step(actions.astype(np.float32))
        practice.add(infos, dones)
        returns += rewards
        totals += [returns[car] for car in np.flatnonzero(dones)]
        returns[dones] = 0

    assert len(practice.runs) == len(totals) >= 4
    assert [run.reward for run in practice.runs] == pytest.approx(totals, abs=1e-3)
    assert all(0 < run.seconds <= 2.0 for run in practice.runs)
    assert {run.end_reason for run in practice.runs} <= {"off_track", "time_limit", "stuck"}


def test_a_summary_becomes_named_numbers_in_groups() -> None:
    runs = [
        RunResult(2.0, 120.0, 0, None, "off_track", 4.0, {"progress": 12.0, "off_track": -10.0}),
        RunResult(30.0, 300.0, 1, 28.0, "time_limit", 10.0, {"progress": 30.0, "off_track": 0.0}),
    ]

    values = scalars("practice", summarize(runs))

    assert values == {
        "practice/score": 16.0,
        "practice/distance": 210.0,
        "practice/average_speed": 30.0,
        "practice/lap_rate": 0.5,
        "practice/best_lap": 28.0,
        "practice_ends/off_track": 0.5,
        "practice_ends/out": 0.0,
        "practice_ends/time_limit": 0.5,
        "practice_ends/stuck": 0.0,
        "practice_reward/progress": 21.0,
        "practice_reward/off_track": -5.0,
    }
    assert "practice/best_lap" not in scalars("practice", summarize(runs[:1]))


def test_a_tests_numbers_include_the_run_from_the_grid() -> None:
    runs = [RunResult(5.0, 50.0, 0, None, "off_track", 2.0, {"progress": 5.0})]
    grid = RunResult(40.0, 400.0, 1, 31.0, "time_limit", 20.0, {"progress": 40.0})

    values = evaluation_scalars(summarize(runs), grid)

    assert values["test/score"] == 5.0
    assert (values["test/grid_score"], values["test/grid_distance"]) == (40.0, 400.0)
    assert values["test/grid_best_lap"] == 31.0
    assert "test/grid_best_lap" not in evaluation_scalars(summarize(runs), runs[0])
