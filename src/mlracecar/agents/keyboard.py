"""A person driving with the keyboard: held keys become smooth steering and a pedal.

Keys are on or off, but a steering wheel isn't, so steering moves towards full lock while a key
is held (taking `STEER_TIME`) and back to straight when it's let go (`CENTRE_TIME`, quicker, as
a wheel self-centres). The pedal follows the keys at once: throttle, brake, or coasting.

The window that owns the keyboard says which keys are held (`KeyboardAgent.keys`); the agent
itself never reads pygame, so it can be tested without a window.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STEER_TIME = 0.15
"""Seconds from straight to full lock while a steering key is held."""

CENTRE_TIME = 0.1
"""Seconds from full lock back to straight after the key is let go."""


@dataclass(frozen=True)
class HeldKeys:
    """Which driving keys are held down."""

    left: bool = False
    right: bool = False
    throttle: bool = False
    brake: bool = False


class KeyboardAgent:
    """Drives one car from the keys held down.

    Args:
        decision_dt: Seconds between two calls to `act` (one driver decision).
    """

    def __init__(self, decision_dt: float) -> None:
        self.decision_dt = decision_dt
        self.keys = HeldKeys()
        """The keys held now; the window updates this before each decision."""
        self._steer = 0.0

    def reset(self, seed: int | None = None) -> None:
        """Straighten the wheel for a new run. The seed is unused: people aren't seeded."""
        self._steer = 0.0

    def act(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """The action for the keys held now, for every row of ``observations`` (which are
        otherwise ignored: the person can see the screen)."""
        target = float(self.keys.left) - float(self.keys.right)
        if target == 0.0 or target * self._steer < 0.0:
            # Let go, or steering the other way: back towards straight, quickly.
            self._steer = _towards(self._steer, 0.0, self.decision_dt / CENTRE_TIME)
        else:
            self._steer = _towards(self._steer, target, self.decision_dt / STEER_TIME)
        pedal = -1.0 if self.keys.brake else 1.0 if self.keys.throttle else 0.0
        action = np.array([self._steer, pedal], dtype=np.float32)
        return np.tile(action, (len(observations), 1))


def _towards(value: float, target: float, step: float) -> float:
    """``value`` moved towards ``target`` by at most ``step``."""
    return value + max(-step, min(step, target - value))
