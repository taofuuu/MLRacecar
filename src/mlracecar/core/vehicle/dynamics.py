"""What every car model shares: the actions, the actuators, and the `DynamicsModel` protocol.

A car model turns each car's action into motion. An action is two numbers in ``[-1, 1]``:

- ``steer``: where the driver turns the wheel; +1 is full lock to the left, -1 to the right.
- ``pedal``: +1 is full throttle, -1 is full braking, 0 is coasting.

The actuators sit between the driver and the physics, as in a real car. The front wheels turn
toward the steering command, but only so fast. The pedal sets the engine's push, which fades at
high speed when the engine's power runs out, or the brakes' push. Air drag and the tyres'
rolling resistance slow the car all the time. There is no reverse gear: braking stops the car
and holds it there.
"""

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.core.vehicle.state import VehicleState

GRAVITY = 9.81
"""m/s²."""
AIR_DENSITY = 1.225
"""kg/m³, at sea level and 15 °C."""

STEER = 0
"""Column of the steering command in an action array."""
PEDAL = 1
"""Column of the pedal in an action array."""


class DynamicsModel(Protocol):
    """Moves N cars of one kind forward by one physics step."""

    @property
    def params(self) -> VehicleParams:
        """The kind of car: its size, engine, brakes, and so on."""
        ...

    def step(self, state: VehicleState, actions: ArrayLike, dt: float) -> VehicleState:
        """The cars' state ``dt`` seconds later, given each car's ``[steer, pedal]`` action."""
        ...


def checked_actions(actions: ArrayLike, cars: int) -> FloatArray:
    """Actions as an ``(N, 2)`` array, clipped to ``[-1, 1]``.

    Raises:
        ValueError: If the shape is wrong or an action isn't a finite number. A driver that
            outputs NaN (a training run gone wrong, say) would otherwise wreck its car's state
            for good without anyone noticing.
    """
    action = np.asarray(actions, dtype=np.float64)
    if action.shape != (cars, 2):
        raise ValueError(
            f"expected [steer, pedal] actions of shape ({cars}, 2), got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("actions must be finite numbers")
    return np.clip(action, -1.0, 1.0)


def steer_toward(
    steer: FloatArray, command: FloatArray, params: VehicleParams, dt: float
) -> FloatArray:
    """The front-wheel angle after ``dt``: it turns toward ``command · max_steer``, by at most
    ``steer_rate · dt``."""
    reach = params.steer_rate * dt
    return steer + np.clip(command * params.max_steer - steer, -reach, reach)


def next_speed(
    speed: FloatArray, pedal: FloatArray, params: VehicleParams, dt: float
) -> FloatArray:
    """The speed after ``dt``, from the engine or the brakes and the air and tyre resistance.

    Resistance and braking only ever slow a car down to a stop, never into reverse.
    """
    # The engine pushes with max_drive_force until power = force · speed reaches max_power
    # (at 25 m/s for the default car), and with max_power / speed above that.
    engine = params.max_power / np.maximum(speed, params.max_power / params.max_drive_force)
    push = np.where(pedal > 0, engine, params.max_brake_force) * pedal
    drag = 0.5 * AIR_DENSITY * params.drag_coefficient * params.frontal_area * speed**2
    rolling = params.rolling_resistance * params.mass * GRAVITY
    result: FloatArray = np.maximum(speed + (push - drag - rolling) / params.mass * dt, 0.0)
    return result
