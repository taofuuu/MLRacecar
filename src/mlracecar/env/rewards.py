"""How the AI is scored: the reward, made of named terms (architecture section 4.7).

Each step every car gets points for the metres it gained along the lap, and loses points for
mistakes such as leaving the road. Each term has its own weight in the `reward` settings
(`mlracecar.config.models.RewardConfig`), and each term's points are reported separately, so
that it's always possible to see which one drove the behaviour.

**Why progress, not speed?** Rewarding speed would reward driving fast anywhere: in circles,
across the grass, or the wrong way. Progress counts only metres gained along the lap
(`race.distance`), so the only way to score is to get round the track; going backwards loses
the points again.
"""

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.config.models import REWARD_TERMS, RewardConfig
from mlracecar.core.geometry import BoolArray, FloatArray
from mlracecar.core.race.events import LapCompleted, OffTrack
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.vehicle.dynamics import checked_actions


class Rewards(NamedTuple):
    """One step's reward for every car, and the points each term gave."""

    total: FloatArray
    """The reward, shape ``(N,)``: the sum of the terms."""
    terms: dict[str, FloatArray]
    """Each term's points (already weighted and signed), shape ``(N,)`` each, in the order of
    `REWARD_TERMS`."""


class RewardFunction:
    """Scores one step of every car.

    Args:
        config: The weight of each term.
        decision_dt: Seconds per step.
    """

    def __init__(self, config: RewardConfig, decision_dt: float) -> None:
        self.config = config
        self.decision_dt = decision_dt

    def __call__(
        self,
        before: Snapshot,
        after: Snapshot,
        actions: ArrayLike,
        previous_actions: ArrayLike,
    ) -> Rewards:
        """The reward for the step from ``before`` to ``after``.

        Args:
            before: The world before the step. It must be the snapshot the step started from:
                a car reset in between would seem to have gained or lost its whole run.
            after: The world after it.
            actions: The ``[steer, pedal]`` each car was given for this step, shape ``(N, 2)``.
            previous_actions: The ones given the step before, shape ``(N, 2)``.

        Raises:
            ValueError: If the actions have the wrong shape or aren't finite numbers.
        """
        count = len(after.cars)
        change = checked_actions(actions, count) - checked_actions(previous_actions, count)
        race = after.race
        left_road = [event.car for event in after.events if isinstance(event, OffTrack)]
        laps = [
            event.car for event in after.events if isinstance(event, LapCompleted) and event.valid
        ]
        # Each term's amount, with its sign: + for what's wanted, - for mistakes.
        amounts = {
            "progress": race.distance - before.race.distance,
            "off_track": -_per_car(left_road, count),
            "time": np.full(count, -self.decision_dt),
            "wrong_way": -self.decision_dt * race.wrong_way,
            "smoothness": -np.einsum("ni,ni->n", change, change),
            "lap": _per_car(laps, count),
        }
        terms = {term: getattr(self.config, term) * amounts[term] for term in REWARD_TERMS}
        total: FloatArray = np.sum(list(terms.values()), axis=0)
        return Rewards(total, terms)


class RewardTally:
    """Adds up each car's points per term over its current run, for reporting.

    Args:
        cars: How many cars.
    """

    def __init__(self, cars: int) -> None:
        self.sums = {term: np.zeros(cars) for term in REWARD_TERMS}
        """Each term's points so far in each car's run, shape ``(N,)`` each."""

    @property
    def total(self) -> FloatArray:
        """Each car's return so far: the sum of all its terms, shape ``(N,)``."""
        result: FloatArray = np.sum(list(self.sums.values()), axis=0)
        return result

    def add(self, rewards: Rewards) -> None:
        """Count one step's rewards."""
        for term, points in rewards.terms.items():
            self.sums[term] += points

    def of(self, car: int) -> dict[str, float]:
        """One car's points per term so far, for an episode's ``info``."""
        return {term: float(points[car]) for term, points in self.sums.items()}

    def clear(self, cars: BoolArray | None = None) -> None:
        """Start counting again for these cars (all of them if ``None``)."""
        for points in self.sums.values():
            points[slice(None) if cars is None else cars] = 0.0


def _per_car(owners: list[int], cars: int) -> FloatArray:
    """How many of these events each car had."""
    counts: FloatArray = np.bincount(np.asarray(owners, dtype=np.int64), minlength=cars).astype(
        np.float64
    )
    return counts
