"""Testing an agent: let it drive from set places, without learning, and see how each run went.

Every test run is one car's run in a `BatchedRacingEnv`, all of them at once, so ten runs take
about as long as one. Each car's run is exactly what a single `RacingEnv` with the same seed
would give, so the results are the same whichever way they're driven, and the same every time
for the same agent and seeds (a deterministic agent drives the same way from the same place).
That's also how `film_run` films a test run: alone, in a `RacingEnv` that draws.

A run is **clean** if the car drove until the time limit without ever leaving the road; the
share of clean runs is the **completion rate**.
"""

from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlracecar.agents.base import Agent
from mlracecar.config.models import REWARD_TERMS, RacecarConfig
from mlracecar.core.track.model import Track
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.env.racing import RacingEnv, ViewerFactory


@dataclass(frozen=True)
class RunResult:
    """How one run went."""

    reward: float
    """The run's total reward: its return."""
    distance: float
    """Metres driven along the track."""
    lap_times: tuple[float, ...]
    """Each valid lap's time in seconds, in the order they were driven."""
    off_tracks: int
    """How many times the car left the road."""
    end_reason: str
    """Why the run ended: ``off_track``, ``out``, ``time_limit``, or ``stuck``."""
    seconds: float
    """How long the run lasted, in seconds of racing."""
    terms: Mapping[str, float]
    """Each reward term's points over the run: what its reward was made of."""

    @property
    def laps(self) -> int:
        """Valid laps completed."""
        return len(self.lap_times)

    @property
    def best_lap(self) -> float | None:
        """The best valid lap time in seconds, or ``None`` without a lap."""
        return min(self.lap_times) if self.lap_times else None

    @property
    def clean(self) -> bool:
        """Whether the car drove until the time limit without ever leaving the road."""
        return self.off_tracks == 0 and self.end_reason == "time_limit"

    @property
    def average_speed(self) -> float:
        """Metres along the track per second, in m/s."""
        return self.distance / self.seconds


class RunTally:
    """Follows each car's run, step by step: how long it has lasted, each valid lap's time, and
    how many times it has left the road. `step` takes each step's ``laps``, ``last_lap``, and
    ``off_track`` (as in the environments' ``info``), one entry per car.

    Args:
        cars: How many cars.
        decision_dt: Seconds of racing per step.
    """

    def __init__(self, cars: int, decision_dt: float) -> None:
        self.decision_dt = decision_dt
        self.steps = np.zeros(cars, dtype=np.int64)
        self.lap_times: list[list[float]] = [[] for _ in range(cars)]
        self.off_tracks = np.zeros(cars, dtype=np.int64)
        self._off = np.zeros(cars, dtype=bool)

    def step(self, laps: ArrayLike, last_lap: ArrayLike, off_track: ArrayLike) -> None:
        """One step of every car."""
        off = np.asarray(off_track, dtype=bool)
        self.steps += 1
        self.off_tracks += off & ~self._off
        self._off = off
        counted = np.array([len(times) for times in self.lap_times])
        for car in np.flatnonzero(np.asarray(laps) > counted).tolist():
            self.lap_times[car].append(float(np.asarray(last_lap)[car]))

    def result(
        self, car: int, reward: float, distance: float, end_reason: str, terms: Mapping[str, float]
    ) -> RunResult:
        """The car's run, which ended at the last step, with what only the end says."""
        return RunResult(
            reward=reward,
            distance=distance,
            lap_times=tuple(self.lap_times[car]),
            off_tracks=int(self.off_tracks[car]),
            end_reason=end_reason,
            seconds=float(self.steps[car] * self.decision_dt),
            terms=terms,
        )

    def restart(self, car: int) -> None:
        """Start following the car's next run."""
        self.steps[car] = 0
        self.lap_times[car] = []
        self.off_tracks[car] = 0
        self._off[car] = False


def drive_test_runs(
    agent: Agent,
    track: Track,
    config: RacecarConfig,
    seeds: Sequence[int],
    start: str = "random",
) -> list[RunResult]:
    """Let ``agent`` drive one run per seed, all at once, and say how each went.

    Args:
        agent: The driver. It sees one observation per run and acts for all of them together.
        track: Where to drive.
        config: Every setting; the same as the agent's environment had in training.
        seeds: One seed per run: where a random start is.
        start: ``"random"`` or ``"grid"``.
    """
    count = len(seeds)
    env = BatchedRacingEnv(track, count, config)
    observations, _ = env.reset(seed=list(seeds), options={"start": start})
    agent.reset()
    tally = RunTally(count, env.timing.decision_dt)
    rewards = np.zeros(count)
    driving = np.ones(count, dtype=bool)
    results: list[RunResult | None] = [None] * count
    while driving.any():
        observations, reward, terminated, truncated, info = env.step(agent.act(observations))
        rewards[driving] += reward[driving]
        tally.step(info["laps"], info["last_lap"], info["off_track"])
        ended = (terminated | truncated) & driving
        for car in np.flatnonzero(ended).tolist():
            results[car] = tally.result(
                car,
                reward=float(rewards[car]),
                distance=float(info["distance"][car]),
                end_reason=str(info["end_reason"][car]),
                terms={term: float(info["episode_terms"][term][car]) for term in REWARD_TERMS},
            )
        driving &= ~ended
    return [result for result in results if result is not None]


def film_run(
    agent: Agent,
    track: Track,
    config: RacecarConfig,
    seed: int,
    start: str,
    viewer: ViewerFactory,
) -> Generator[NDArray[np.uint8], None, None]:
    """The pictures of the test run that `drive_test_runs` drives with ``seed`` and ``start``:
    one at the start and one after each step, until the run ends.

    They're made one at a time, as they're asked for, so a long run doesn't have to fit in
    memory: a minute of racing is 1,201 pictures. Closing it early closes the environment.

    Args:
        agent: The driver.
        track: Where to drive.
        config: Every setting; the same as the agent's environment had in training.
        seed: Where a random start is.
        start: ``"random"`` or ``"grid"``.
        viewer: Makes the viewer that draws the pictures.
    """
    env = RacingEnv(track, config, render_mode="rgb_array", viewer=viewer)
    try:
        observation, _ = env.reset(seed=seed, options={"start": start})
        agent.reset()
        while True:
            yield _picture(env)
            observation, _, terminated, truncated, _ = env.step(agent.act(observation[None])[0])
            if terminated or truncated:
                yield _picture(env)
                return
    finally:
        env.close()


def _picture(env: RacingEnv) -> NDArray[np.uint8]:
    picture = env.render()
    assert picture is not None  # drawn as pictures, not in a window
    return picture


def summarize(results: Sequence[RunResult]) -> dict[str, Any]:
    """Runs in a few numbers: the mean reward (the score), the completion rate (the share of
    clean runs), the mean distance and speed, the laps, the share of runs with a lap, the mean
    and best lap, the times the cars left the road, how the runs ended, and each reward term's
    mean points per run."""
    laps = [time for result in results for time in result.lap_times]
    return {
        "runs": len(results),
        "score": float(np.mean([result.reward for result in results])),
        "completion_rate": float(np.mean([result.clean for result in results])),
        "distance": float(np.mean([result.distance for result in results])),
        "average_speed": float(np.mean([result.average_speed for result in results])),
        "laps": len(laps),
        "lap_rate": float(np.mean([result.laps > 0 for result in results])),
        "mean_lap": float(np.mean(laps)) if laps else None,
        "best_lap": min(laps) if laps else None,
        "off_tracks": sum(result.off_tracks for result in results),
        "end_reasons": dict(sorted(Counter(result.end_reason for result in results).items())),
        "terms": {
            term: float(np.mean([result.terms[term] for result in results]))
            for term in results[0].terms
        },
    }


def as_dict(result: RunResult) -> dict[str, Any]:
    """One run's result as plain JSON-ready data, with its laps, best lap, whether it was clean,
    and its average speed."""
    return asdict(result) | {
        "lap_times": list(result.lap_times),
        "terms": dict(result.terms),
        "laps": result.laps,
        "best_lap": result.best_lap,
        "clean": result.clean,
        "average_speed": result.average_speed,
    }
