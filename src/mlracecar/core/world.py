"""The simulation world: cars on a track, advanced in fixed time steps (architecture section 4.4).

So far this holds the timing the world runs on.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Timing:
    """How simulated time moves forward: physics in fixed steps, drivers deciding every few steps.

    Fixed steps make a run come out the same on any computer, however fast it is. Each driver
    decision (an action) is held for `action_repeat` physics steps.
    """

    physics_hz: int
    """Physics steps per simulated second."""
    action_repeat: int
    """Physics steps per driver decision."""

    @property
    def dt(self) -> float:
        """Length of one physics step in seconds."""
        return 1.0 / self.physics_hz

    @property
    def decision_dt(self) -> float:
        """Simulated seconds between driver decisions."""
        return self.action_repeat / self.physics_hz
