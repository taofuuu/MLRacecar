"""When a car's run (episode) ends while the AI trains, and how (architecture section 4.8).

A run ends in one of two ways, and Gymnasium keeps them apart because learning depends on it:

- **Terminated:** the run really is over, and nothing can follow. The car left the road (with
  ``episode.end_off_track``), or the race rules took it out (``race.off_track: terminate``).
- **Truncated:** we stopped it, although the car could have carried on: the time limit, or it
  was stuck, slower than `STUCK_SPEED` for too long. Learning then still counts on what would
  have followed.

The `episode` settings (`mlracecar.config.models.EpisodeConfig`) set the rules. The race's own
off-track rule still applies while the car is on the grass; ending the run here is what keeps
the AI from learning that the grass is a shortcut, whatever `race.off_track` says.
"""

import math
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from mlracecar.config.models import EpisodeConfig
from mlracecar.core.geometry import BoolArray, IntArray
from mlracecar.core.race.events import OffTrack
from mlracecar.core.snapshot import Snapshot

STUCK_SPEED = 1.0
"""A car slower than this, in m/s, is getting nowhere."""


class EndReason(StrEnum):
    """Why a car's run ended."""

    OFF_TRACK = "off_track"
    """Its centre left the road (terminated)."""
    OUT = "out"
    """The race rules took it out (terminated)."""
    TIME_LIMIT = "time_limit"
    """It ran out of time (truncated)."""
    STUCK = "stuck"
    """It was slower than `STUCK_SPEED` for too long (truncated)."""


class Endings(NamedTuple):
    """Whose runs ended in a step, and why; one entry per car."""

    terminated: BoolArray
    """The run is really over; nothing can follow."""
    truncated: BoolArray
    """The run was stopped, but could have gone on. Never set together with ``terminated``."""
    reasons: tuple[EndReason | None, ...]
    """Why each run ended, or ``None`` for a car still going."""


class EpisodeRules:
    """Keeps every car's run clock and decides when each run ends.

    Args:
        config: The rules: leaving the road, the time limit, and how long being stuck may last.
        decision_dt: Seconds per step.
        cars: How many cars.
    """

    def __init__(self, config: EpisodeConfig, decision_dt: float, cars: int) -> None:
        self.config = config
        # Whole steps, so that the limits don't depend on rounding: 60 s is exactly 1200 steps.
        self.max_steps = math.ceil(config.time_limit / decision_dt - 1e-9)
        """The most steps a run lasts."""
        self.max_stuck_steps = math.ceil(config.stuck_time / decision_dt - 1e-9)
        """The most steps in a row a car may be stuck."""
        self.steps: IntArray = np.zeros(cars, dtype=np.int64)
        """Steps each car's run has lasted so far."""
        self.stuck_steps: IntArray = np.zeros(cars, dtype=np.int64)
        """Steps in a row each car has been stuck."""

    def start(self, cars: BoolArray | None = None) -> None:
        """Start the clocks again for these cars (all of them if ``None``), as their runs begin."""
        chosen = slice(None) if cars is None else cars
        self.steps[chosen] = 0
        self.stuck_steps[chosen] = 0

    def check(self, after: Snapshot) -> Endings:
        """Count one more step for every car, and say whose runs ended in it.

        Args:
            after: The world after the step.
        """
        cars, race = after.cars, after.race
        self.steps += 1
        stuck = cars.speed < STUCK_SPEED
        self.stuck_steps = np.where(stuck, self.stuck_steps + 1, 0)

        left_road = np.zeros(len(cars), dtype=bool)
        left_road[[event.car for event in after.events if isinstance(event, OffTrack)]] = True
        off_track = (left_road | race.off_track) & self.config.end_off_track
        out_of_time = self.steps >= self.max_steps
        too_long_stuck = self.stuck_steps >= self.max_stuck_steps

        terminated = off_track | race.out
        truncated = (out_of_time | too_long_stuck) & ~terminated
        reasons = tuple(
            _reason(off_track[car], race.out[car], out_of_time[car], too_long_stuck[car])
            for car in range(len(cars))
        )
        return Endings(terminated, truncated, reasons)


def _reason(off_track: bool, out: bool, out_of_time: bool, stuck: bool) -> EndReason | None:
    """The first reason that applies: endings before stops, as for the flags."""
    if off_track:
        return EndReason.OFF_TRACK
    if out:
        return EndReason.OUT
    if out_of_time:
        return EndReason.TIME_LIMIT
    if stuck:
        return EndReason.STUCK
    return None
