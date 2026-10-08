"""Tests for mlracecar.training.evaluation: driving test runs and summing them up."""

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from mlracecar.config.models import REWARD_TERMS, EpisodeConfig, RacecarConfig
from mlracecar.core.geometry import FloatArray
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.env.racing import RacingEnv, load_track
from mlracecar.training.evaluation import (
    RunResult,
    as_dict,
    drive_test_runs,
    film_run,
    summarize,
)

TECHNICAL = load_track(Path(__file__).parents[3] / "tracks" / "technical.json")
ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 48)
SHORT = RacecarConfig(episode=EpisodeConfig(time_limit=6.0))


class Steady:
    """Drives every car with the same action; counts its resets."""

    def __init__(self, steer: float, pedal: float) -> None:
        self.action = np.array([steer, pedal], dtype=np.float32)
        self.resets = 0

    def reset(self, seed: int | None = None) -> None:
        self.resets += 1

    def act(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.tile(self.action, (len(observations), 1))


def alone(track: Track, config: RacecarConfig, seed: int, start: str, agent: Steady) -> RunResult:
    """The same run driven in a single environment."""
    env = RacingEnv(track, config)
    observation, _ = env.reset(seed=seed, options={"start": start})
    total, steps = 0.0, 0
    while True:
        observation, reward, terminated, truncated, info = env.step(agent.act(observation[None])[0])
        total += reward
        steps += 1
        if terminated or truncated:
            return RunResult(
                total,
                info["distance"],
                info["laps"],
                info["best_lap"],
                info["end_reason"],
                steps * env.timing.decision_dt,
                info["episode_terms"],
            )


@pytest.mark.parametrize("start", ["random", "grid"])
def test_each_run_is_exactly_what_a_single_environment_gives(start: str) -> None:
    agent = Steady(0.05, 0.4)
    seeds = [11, 12, 13, 14]

    results = drive_test_runs(agent, TECHNICAL, SHORT, seeds, start)

    assert results == [alone(TECHNICAL, SHORT, seed, start, agent) for seed in seeds]
    assert agent.resets == 1


def test_runs_end_for_different_reasons_and_each_is_recorded_once() -> None:
    # Straight on at full throttle leaves the round track; standing still gets stuck.
    off = drive_test_runs(Steady(0.0, 1.0), CIRCLE, RacecarConfig(), [1], "grid")[0]
    stuck = drive_test_runs(Steady(0.0, 0.0), CIRCLE, RacecarConfig(), [1], "grid")[0]

    assert off.end_reason == "off_track"
    assert off.distance > 0
    assert stuck.end_reason == "stuck"
    assert stuck.seconds == pytest.approx(5.0)  # the stuck rule
    assert stuck.distance == 0.0
    assert stuck.average_speed == 0.0


def test_a_lap_driven_in_the_run_is_counted_with_its_time() -> None:
    # A gentle left round the circle at a steady pace, from the grid, for two minutes.
    config = RacecarConfig(episode=EpisodeConfig(time_limit=120.0))

    result = drive_test_runs(_CircleDriver(), CIRCLE, config, [0], "grid")[0]

    assert result.laps >= 1
    assert result.best_lap is not None
    assert result.best_lap == pytest.approx(2 * math.pi * 60 / 20, rel=0.1)


class _CircleDriver:
    """Holds about 20 m/s round the 60 m circle: steers by the observation's heading input."""

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        speed = observations[:, 15] * 100  # the speed input, in m/s
        offset = observations[:, 18]  # the offset input, left positive
        steer = np.clip(0.33 - 0.5 * offset, -1, 1)
        pedal = np.clip((20 - speed) * 0.3, -1, 1)
        return np.column_stack([steer, pedal]).astype(np.float32)


def test_a_runs_reward_is_the_sum_of_its_terms() -> None:
    [result] = drive_test_runs(Steady(0.05, 0.4), TECHNICAL, SHORT, [3])

    assert set(result.terms) == set(REWARD_TERMS)
    assert sum(result.terms.values()) == pytest.approx(result.reward)
    assert result.terms["progress"] > 0


class Pictures:
    """A viewer that draws tiny pictures: each one shows how far the car has driven."""

    made: list["Pictures"] = []  # noqa: RUF012  # every one made, to look at after

    def __init__(self, track: Track, car: VehicleParams, mode: str, fps: float) -> None:
        assert mode == "rgb_array"
        self.closed = False
        Pictures.made.append(self)

    def render(self, snapshot: Snapshot, rays: FloatArray | None) -> NDArray[np.uint8]:
        distance = min(int(snapshot.race.distance[0]), 255)
        return np.full((4, 6, 3), distance, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("start", ["random", "grid"])
def test_a_filmed_run_is_the_test_run_with_the_same_seed(start: str) -> None:
    agent = Steady(0.05, 0.4)
    [result] = drive_test_runs(agent, TECHNICAL, SHORT, [7], start)
    Pictures.made.clear()

    frames = list(film_run(agent, TECHNICAL, SHORT, 7, start, Pictures))

    assert len(frames) == round(result.seconds / 0.05) + 1  # the start, then every step
    assert frames[0].shape == (4, 6, 3)
    assert frames[-1][0, 0, 0] == min(int(result.distance), 255)
    [viewer] = Pictures.made
    assert viewer.closed


def test_a_film_is_made_as_its_pictures_are_asked_for() -> None:
    Pictures.made.clear()
    frames = film_run(Steady(0.0, 0.4), TECHNICAL, SHORT, 7, "grid", Pictures)
    assert Pictures.made == []  # nothing driven yet

    first = next(frames)
    frames.close()  # stopped early: the environment is closed all the same

    assert first.shape == (4, 6, 3)
    assert Pictures.made[0].closed


def points(progress: float, off_track: float) -> dict[str, float]:
    return {"progress": progress, "off_track": off_track}


def test_the_summary_gives_the_score_and_how_the_runs_went() -> None:
    results = [
        RunResult(10.0, 100.0, 1, 50.0, "time_limit", 10.0, points(10.0, 0.0)),
        RunResult(-6.0, 40.0, 0, None, "off_track", 4.0, points(4.0, -10.0)),
        RunResult(2.0, 20.0, 2, 45.0, "time_limit", 10.0, points(2.0, 0.0)),
    ]

    summary = summarize(results)

    assert summary == {
        "runs": 3,
        "score": 2.0,
        "distance": pytest.approx(160 / 3),
        "average_speed": pytest.approx((10 + 10 + 2) / 3),
        "laps": 3,
        "lap_rate": pytest.approx(2 / 3),
        "best_lap": 45.0,
        "end_reasons": {"off_track": 1, "time_limit": 2},
        "terms": {"progress": pytest.approx(16 / 3), "off_track": pytest.approx(-10 / 3)},
    }
    assert summarize(results[1:2])["best_lap"] is None
    assert summarize(results[1:2])["lap_rate"] == 0.0


def test_a_result_as_plain_data_includes_its_average_speed() -> None:
    result = RunResult(1.0, 30.0, 0, None, "stuck", 6.0, points(1.0, 0.0))

    assert as_dict(result) == {
        "reward": 1.0,
        "distance": 30.0,
        "laps": 0,
        "best_lap": None,
        "end_reason": "stuck",
        "seconds": 6.0,
        "terms": {"progress": 1.0, "off_track": 0.0},
        "average_speed": 5.0,
    }
