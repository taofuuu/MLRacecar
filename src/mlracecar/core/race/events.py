"""What happens in a race, as plain data: easy to log, test, and replay (architecture 4.5)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LapCompleted:
    """A car crossed the start/finish line at the end of a lap."""

    car: int
    """Which car."""
    time: float
    """The lap time in seconds."""
    sectors: tuple[float, ...]
    """The time spent in each sector, in seconds; NaN for a sector whose start was missed."""
    valid: bool
    """Whether the car crossed every checkpoint in order. Only valid laps count."""
    at: float
    """The world's time when the car crossed the line."""


type RaceEvent = LapCompleted
"""Anything that can happen in a race."""
