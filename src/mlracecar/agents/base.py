"""The `Agent` protocol: anything that drives cars (architecture section 4.9, ADR-0006).

A person at the keyboard, a trained policy, or a scripted test driver all look the same to the
rest of MLRacecar: they get observations and return ``[steer, pedal]`` actions.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Agent(Protocol):
    """Maps observations to actions, a batch of cars at a time."""

    def reset(self, seed: int | None = None) -> None:
        """Get ready for a new run: forget what happened before, and reseed any randomness."""
        ...

    def act(self, observations: NDArray[np.float32]) -> NDArray[np.float32]:
        """Actions for a batch of observations: shape ``(n, obs_dim)`` in, ``(n, 2)`` out.

        Each action is ``[steer, pedal]`` in ``[-1, 1]``: steer +1 is full left, pedal +1 is
        full throttle and -1 full braking.
        """
        ...
