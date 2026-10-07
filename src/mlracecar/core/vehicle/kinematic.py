"""The kinematic bicycle: simple, stable car physics in which the car goes where its wheels point.

Each pair of wheels acts as one, like a bicycle. Without tyre slip the rear wheels roll straight
ahead, so the car turns about a point level with the rear axle, ``wheelbase / tan(steer)`` from
it, and the centre moves at a small angle to the heading (the sideslip angle).

Real tyres can only hold so much sideways force, so here the turn is limited by grip: at speed
``v`` the car can't follow a tighter circle than ``v² / (grip · g)``. Steering harder than that
turns no tighter and the car runs wide (understeer), so it has to slow down for corners
(ADR-0014).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.vehicle.dynamics import (
    GRAVITY,
    PEDAL,
    STEER,
    checked_actions,
    next_speed,
    steer_toward,
)
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.core.vehicle.state import VehicleState


@dataclass(frozen=True)
class KinematicBicycle:
    """The kinematic bicycle model for N cars of one kind (architecture section 4.3)."""

    params: VehicleParams

    def step(self, state: VehicleState, actions: ArrayLike, dt: float) -> VehicleState:
        """The cars' state ``dt`` seconds later.

        Semi-implicit Euler: the new steering angle and speed come first, and the turn and the
        move use them.

        Args:
            state: N cars.
            actions: ``[steer, pedal]`` for each car, shape ``(N, 2)``, clipped to ``[-1, 1]``.
            dt: Step length in seconds.

        Raises:
            ValueError: If the actions have the wrong shape or aren't finite numbers.
        """
        params = self.params
        action = checked_actions(actions, len(state))
        steer = steer_toward(state.steer, action[:, STEER], params, dt)
        speed = next_speed(state.speed, action[:, PEDAL], params, dt)

        tan_turn = np.tan(self._turn(steer, speed))
        # The centre sits halfway along the wheelbase, so it moves at the sideslip angle
        # atan(tan(turn) / 2) to the heading.
        sideslip = np.arctan(0.5 * tan_turn)
        yaw_rate = speed * np.cos(sideslip) * tan_turn / params.wheelbase
        yaw = wrap_angle(state.yaw + yaw_rate * dt)
        course = yaw + sideslip
        return VehicleState(
            x=state.x + speed * np.cos(course) * dt,
            y=state.y + speed * np.sin(course) * dt,
            yaw=yaw,
            vx=speed * np.cos(sideslip),
            vy=speed * np.sin(sideslip),
            yaw_rate=yaw_rate,
            steer=steer,
        )

    def _turn(self, steer: FloatArray, speed: FloatArray) -> FloatArray:
        """The wheel angle the tyres can follow at this speed: the steering, limited by grip.

        Turning about a point ``wheelbase / tan(angle)`` from the rear axle at speed ``v`` takes
        a sideways acceleration of at most ``v² · tan(angle) / wheelbase``, and the tyres give
        ``grip · g``.
        """
        params = self.params
        limit = np.arctan2(params.grip * GRAVITY * params.wheelbase, speed * speed)
        return np.clip(steer, -limit, limit)
