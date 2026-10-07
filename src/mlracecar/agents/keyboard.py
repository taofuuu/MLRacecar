"""A person driving with the keyboard: held keys become smooth steering and a pedal.

Keys are on or off, but a steering wheel isn't, so steering moves towards full turn while a key
is held (taking `STEER_TIME`) and back to straight when it's let go (`CENTRE_TIME`, quicker, as
a wheel self-centres). The pedal follows the keys at once: throttle, brake, or coasting.

Steering is speed-sensitive, as in most keyboard racing games. At speed the tyres can only
follow a small wheel angle (about 2 degrees at 100 km/h), so a key that turned the wheels to
full lock would make every tap the hardest turn the car can make. Instead, a key turns the
wheels only as far as the tyres can use, plus `GRIP_MARGIN`: holding one turns as hard as the
car can, a short tap makes a small correction, and letting go straightens the car at once.
Below about 25 km/h the keys still reach full lock, for hairpins.

The window that owns the keyboard says which keys are held (`KeyboardAgent.keys`); the agent
itself never reads pygame, so it can be tested without a window.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mlracecar.core.vehicle.dynamics import GRAVITY
from mlracecar.core.vehicle.params import VehicleParams

STEER_TIME = 0.15
"""Seconds from straight to a full turn while a steering key is held."""

CENTRE_TIME = 0.1
"""Seconds from a full turn back to straight after the key is let go."""

GRIP_MARGIN = 1.1
"""How far past what the tyres can follow a held key turns the wheels at speed."""


@dataclass(frozen=True)
class HeldKeys:
    """Which driving keys are held down."""

    left: bool = False
    right: bool = False
    throttle: bool = False
    brake: bool = False


class KeyboardAgent:
    """Drives cars from the keys held down.

    Args:
        decision_dt: Seconds between two calls to `act` (one driver decision).
        car: The car being driven, for speed-sensitive steering; without it, the keys always
            reach full lock.
    """

    def __init__(self, decision_dt: float, car: VehicleParams | None = None) -> None:
        self.decision_dt = decision_dt
        self.car = car
        self.keys = HeldKeys()
        """The keys held now; the window updates this before each decision."""
        self._turn = 0.0  # how far the keys have turned the wheel, as a share of their reach

    def reset(self, seed: int | None = None) -> None:
        """Straighten the wheel for a new run. The seed is unused: people aren't seeded."""
        self._turn = 0.0

    def act(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """The action for the keys held now, for each car.

        Args:
            observations: One row per car. Column 0, if there is one, is the car's speed in m/s,
                for speed-sensitive steering. The person sees the rest on the screen.
        """
        target = float(self.keys.left) - float(self.keys.right)
        if target == 0.0 or target * self._turn < 0.0:
            # Let go, or steering the other way: back towards straight, quickly.
            self._turn = _towards(self._turn, 0.0, self.decision_dt / CENTRE_TIME)
        else:
            self._turn = _towards(self._turn, target, self.decision_dt / STEER_TIME)
        pedal = -1.0 if self.keys.brake else 1.0 if self.keys.throttle else 0.0
        speed = observations[:, 0] if observations.shape[1] else np.zeros(len(observations))
        steer = self._turn * self.reach(speed.astype(np.float64))
        return np.column_stack([steer, np.full(len(steer), pedal)]).astype(np.float32)

    def reach(self, speed: NDArray[np.float64]) -> NDArray[np.float64]:
        """How far a held key turns the wheels at each speed (m/s), as a share of full lock."""
        if self.car is None:
            return np.ones_like(speed)
        usable = np.arctan2(self.car.grip * GRAVITY * self.car.wheelbase, speed * speed)
        result: NDArray[np.float64] = np.minimum(1.0, GRIP_MARGIN * usable / self.car.max_steer)
        return result


def _towards(value: float, target: float, step: float) -> float:
    """``value`` moved towards ``target`` by at most ``step``."""
    return value + max(-step, min(step, target - value))
