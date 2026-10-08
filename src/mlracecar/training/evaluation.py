"""Testing an agent: let it drive from set places, without learning, and see how each run went.

Every test run is one car's run in a `BatchedRacingEnv`, all of them at once, so ten runs take
about as long as one. Each car's run is exactly what a single `RacingEnv` with the same seed
would give, so the results are the same whichever way they're driven, and the same every time
for the same agent and seeds (a deterministic agent drives the same way from the same place).
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mlracecar.agents.base import Agent
from mlracecar.config.models import RacecarConfig
from mlracecar.core.track.model import Track
from mlracecar.env.batched import BatchedRacingEnv


@dataclass(frozen=True)
class RunResult:
    """How one test run went."""

    reward: float
    """The run's total reward: its return."""
    distance: float
    """Metres driven along the track."""
    laps: int
    """Valid laps completed."""
    best_lap: float | None
    """The best valid lap time in seconds, or ``None`` without a lap."""
    end_reason: str
    """Why the run ended: ``off_track``, ``out``, ``time_limit``, or ``stuck``."""
    seconds: float
    """How long the run lasted, in seconds of racing."""

    @property
    def average_speed(self) -> float:
        """Metres along the track per second, in m/s."""
        return self.distance / self.seconds


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
    rewards = np.zeros(count)
    steps = np.zeros(count, dtype=np.int64)
    driving = np.ones(count, dtype=bool)
    results: list[RunResult | None] = [None] * count
    while driving.any():
        observations, reward, terminated, truncated, info = env.step(agent.act(observations))
        rewards[driving] += reward[driving]
        steps[driving] += 1
        ended = (terminated | truncated) & driving
        for car in np.flatnonzero(ended):
            best = float(info["best_lap"][car])
            results[car] = RunResult(
                reward=float(rewards[car]),
                distance=float(info["distance"][car]),
                laps=int(info["laps"][car]),
                best_lap=best if np.isfinite(best) else None,
                end_reason=str(info["end_reason"][car]),
                seconds=float(steps[car] * env.timing.decision_dt),
            )
        driving &= ~ended
    return [result for result in results if result is not None]


def summarize(results: Sequence[RunResult]) -> dict[str, Any]:
    """The test runs in a few numbers: the mean reward (the score), distance, and speed, the
    laps and the best lap, and how the runs ended."""
    laps = [result.best_lap for result in results if result.best_lap is not None]
    return {
        "runs": len(results),
        "score": float(np.mean([result.reward for result in results])),
        "distance": float(np.mean([result.distance for result in results])),
        "average_speed": float(np.mean([result.average_speed for result in results])),
        "laps": sum(result.laps for result in results),
        "best_lap": min(laps) if laps else None,
        "end_reasons": dict(sorted(Counter(result.end_reason for result in results).items())),
    }


def as_dict(result: RunResult) -> dict[str, Any]:
    """One run's result as plain JSON-ready data, with its average speed."""
    return asdict(result) | {"average_speed": result.average_speed}
