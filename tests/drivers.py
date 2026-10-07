"""Test-only drivers: scripted cars that drive a track by themselves."""

import numpy as np

from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.dynamics import GRAVITY
from mlracecar.core.vehicle.params import VehicleParams


class CenterlineDriver:
    """Follows the centerline at a steady speed, braking in time for bends.

    Steering is pure pursuit: each car aims at the centerline ``lookahead`` metres ahead of where
    it is along the lap and steers onto the arc that passes through that point. The pedal holds
    the target speed, which drops ahead of each bend to what the tyres can take there, using
    ``corner_grip`` of their grip, and braking at ``braking`` m/s² on the way in.
    """

    def __init__(
        self,
        track: Track,
        params: VehicleParams,
        speed: float,
        *,
        lookahead: float = 5.0,
        gain: float = 2.0,
        corner_grip: float = 0.8,
        braking: float = 0.5 * GRAVITY,
    ) -> None:
        self.track = track
        self.params = params
        self.speed = speed
        self.lookahead = lookahead
        self.gain = gain
        self.braking = braking
        self._corner_acceleration = corner_grip * params.grip * GRAVITY
        self._ahead = np.arange(0.0, speed**2 / (2 * braking) + 10.0, 2.0)

    def act(self, snapshot: Snapshot) -> FloatArray:
        """``[steer, pedal]`` for every car in the snapshot, shape ``(N, 2)``."""
        cars, race = snapshot.cars, snapshot.race
        target = self.track.pose_at(race.arc_length + self.lookahead).position - cars.position
        course = cars.yaw + np.arctan2(cars.vy, cars.vx)  # where the car is actually going
        angle = wrap_angle(np.arctan2(target[:, 1], target[:, 0]) - course)
        curvature = 2 * np.sin(angle) / np.hypot(target[:, 0], target[:, 1])
        steer = np.arctan(self.params.wheelbase * curvature) / self.params.max_steer
        pedal = self.gain * (self._target_speed(race.arc_length) - cars.speed)
        return np.clip(np.column_stack([steer, pedal]), -1.0, 1.0)

    def _target_speed(self, arc_length: FloatArray) -> FloatArray:
        """The fastest speed from which every bend in braking range can still be taken."""
        line = self.track.centerline
        spots = arc_length[:, None] + self._ahead
        bend = np.abs(np.interp(spots, line.arc_length, line.curvature, period=line.length))
        corner_speed = np.sqrt(self._corner_acceleration / np.maximum(bend, 1e-9))
        allowed = np.sqrt(corner_speed**2 + 2 * self.braking * self._ahead)
        result: FloatArray = np.minimum(self.speed, allowed.min(axis=1))
        return result
