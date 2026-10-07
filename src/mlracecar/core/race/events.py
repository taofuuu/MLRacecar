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


@dataclass(frozen=True)
class OffTrack:
    """A car's centre left the road."""

    car: int
    """Which car."""
    arc_length: float
    """Where along the lap, in metres."""
    at: float
    """The world's time at the end of the update in which it happened."""


@dataclass(frozen=True)
class WrongWay:
    """A car started driving the wrong way round the track."""

    car: int
    """Which car."""
    arc_length: float
    """Where along the lap, in metres."""
    at: float
    """The world's time at the end of the update in which it happened."""


type RaceEvent = LapCompleted | OffTrack | WrongWay
"""Anything that can happen in a race."""
