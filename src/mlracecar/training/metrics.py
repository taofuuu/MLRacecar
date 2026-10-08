"""The racing numbers a training run tracks: how the practice runs and the tests are going.

Practice runs are the cars' runs while the agent learns (it acts with some randomness, to
explore); test runs are the ones it drives without learning (`evaluation.drive_test_runs`).
Both are summed up the same way (`evaluation.summarize`) and named the same way, with
``practice`` or ``test`` as the group:

- ``<group>/score``, ``/completion_rate``, ``/distance``, ``/average_speed``, ``/lap_rate``,
  ``/mean_lap``, ``/best_lap``: as in `summarize` (the lap times only once there is a lap);
- ``<group>_ends/<reason>``: the share of runs that ended that way (``off_track``, ...);
- ``<group>_reward/<term>``: each reward term's mean points per run.

Tests add the run from the grid: ``test/grid_score``, ``/grid_distance``, ``/grid_best_lap``.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlracecar.config.models import REWARD_TERMS
from mlracecar.env.episodes import EndReason
from mlracecar.training.evaluation import RunResult, RunTally, summarize

PRACTICE_WINDOW = 100
"""Practice runs summed up: the latest this many, as Stable-Baselines3 does for its own."""


class PracticeRuns:
    """The cars' latest practice runs, collected from each training step's ``infos``.

    Args:
        decision_dt: Seconds of racing per step.
        window: How many of the latest runs to keep.
    """

    def __init__(self, decision_dt: float, window: int = PRACTICE_WINDOW) -> None:
        self.decision_dt = decision_dt
        self.runs: deque[RunResult] = deque(maxlen=window)
        self._tally: RunTally | None = None

    def add(self, infos: Sequence[Mapping[str, Any]], dones: NDArray[np.bool_]) -> None:
        """One step of every car (Stable-Baselines3's ``infos`` and ``dones``): keep the runs
        that ended in it. Every car must have started a fresh run before the first step."""
        if self._tally is None:
            self._tally = RunTally(len(infos), self.decision_dt)
        tally = self._tally
        tally.step(
            [info["laps"] for info in infos],
            [info["last_lap"] for info in infos],
            [info["off_track"] for info in infos],
        )
        for car in np.flatnonzero(dones).tolist():
            info = infos[car]
            terms = {term: float(info["episode_terms"][term]) for term in REWARD_TERMS}
            self.runs.append(
                tally.result(
                    car,
                    reward=sum(terms.values()),
                    distance=float(info["distance"]),
                    end_reason=str(info["end_reason"]),
                    terms=terms,
                )
            )
            tally.restart(car)

    def scalars(self) -> dict[str, float]:
        """The latest runs as named numbers; none before the first run ends."""
        return scalars("practice", summarize(self.runs)) if self.runs else {}


def scalars(group: str, summary: Mapping[str, Any]) -> dict[str, float]:
    """A summary from `summarize` as named numbers for a tracker."""
    values = {
        f"{group}/{key}": float(summary[key])
        for key in ("score", "completion_rate", "distance", "average_speed", "lap_rate")
    }
    for lap in ("mean_lap", "best_lap"):
        if summary[lap] is not None:
            values[f"{group}/{lap}"] = float(summary[lap])
    for reason in EndReason:
        ended = summary["end_reasons"].get(reason.value, 0)
        values[f"{group}_ends/{reason.value}"] = ended / summary["runs"]
    for term, points in summary["terms"].items():
        values[f"{group}_reward/{term}"] = float(points)
    return values


def evaluation_scalars(summary: Mapping[str, Any], grid: RunResult) -> dict[str, float]:
    """A test's summary and its run from the grid as named numbers for a tracker."""
    values = scalars("test", summary)
    values |= {"test/grid_score": grid.reward, "test/grid_distance": grid.distance}
    if grid.best_lap is not None:
        values["test/grid_best_lap"] = grid.best_lap
    return values
