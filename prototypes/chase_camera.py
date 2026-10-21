"""PROTOTYPE for #114, kept as a record and never merged: compare chase-camera styles by driving.

The verdict: "slow, catches up" felt best (turn time 4.5 s, quick past 60 degrees, plus
looking ahead that can be switched off); it was built for real in mlracecar.render.race. The
measurements behind it are on #114. Run on the branch it was written for (feat/114-chase-camera's
parent, with the arcade car):

    uv run python prototypes/chase_camera.py tracks/experiment-1-track.json playground/arcade.yaml

The first argument is the track; any others are settings files. Press V to switch style:
follow (today, north up), locked to the car, eased, eased + look-ahead, follows the road,
quarter turns, calm quarter turns, slow + catches up (the default: eased over 4.5 s, but quick
once the car points more than 60 deg off, so left and right rarely swap).
[ makes the eased styles turn quicker, ] slower and calmer. The race window's usual keys still work
(arrows/WASD, R restart, C camera, 1-4 overlays, wheel zoom).

Known gap: the start-line chevron doesn't turn with the view.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
from numpy.typing import ArrayLike

from mlracecar.config.files import load_config
from mlracecar.config.models import RacecarConfig
from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.io.track_file import read_track_file
from mlracecar.play.drive import DriveWindow
from mlracecar.render.camera import Camera
from mlracecar.render.race import CameraMode, RaceRenderer

STYLES = (
    "follow (today, north up)",
    "locked to the car",
    "eased",
    "eased + look-ahead",
    "follows the road",
    "quarter turns",
    "calm quarter turns",
    "slow, catches up",
)
EASE = 0.3
"""Seconds the eased styles take to turn most of the way (63%) to where they aim; [ and ]."""
ROAD_AHEAD = 10.0
"""Metres ahead of the car where the road-following style reads the road's direction."""
QUARTER_HOLD = math.radians(60)
"""The quarter-turn styles hold their view until the road points this far from straight up."""
SWING_TIME = 1.0
"""Seconds a calm quarter turn takes, starting and ending gently; [ and ] change it."""
STRAIGHT_AHEAD = 1.0
"""Seconds of road ahead that must be nearly straight before a calm quarter turn starts."""
STRAIGHT_BEND = math.radians(15)
"""How much that road may bend and still count as straight."""
MUST_SWING = math.radians(100)
"""A calm quarter turn starts even in a corner once the road points this far from straight up,
so the car never drives down the screen for long."""
SLOW_EASE = 4.5
"""Turn time of the slow style ("slow, catches up"), the one that felt best: [ and ] change it."""
CATCH_UP = math.radians(60)
"""The slow style turns quicker for any part of the angle past this, so the car rarely points
down the screen (left and right swapped)."""
CATCH_UP_TIME = 0.5
"""Turn time for the part past `CATCH_UP`."""
LOOK_AHEAD = 0.25
"""Share of the window height the car sits below the centre, in the look-ahead style."""


@dataclass(frozen=True)
class TurnedCamera(Camera):
    """A camera that can turn: ``up`` is the world direction shown pointing up the window."""

    up: float = math.pi / 2

    def to_screen(self, points: ArrayLike) -> FloatArray:
        relative = np.asarray(points, dtype=np.float64) - self.center
        turned = _turn(relative, math.pi / 2 - self.up)
        return turned * self.scale * (1.0, -1.0) + np.asarray(self.size) / 2

    def to_world(self, pixels: ArrayLike) -> FloatArray:
        screen = np.asarray(pixels, dtype=np.float64) - np.asarray(self.size) / 2
        return _turn(screen * (1.0, -1.0) / self.scale, self.up - math.pi / 2) + self.center

    def visible_area(self) -> tuple[FloatArray, FloatArray]:
        width, height = self.size
        corners = self.to_world([(0, 0), (width, 0), (0, height), (width, height)])
        return corners.min(axis=0), corners.max(axis=0)


def _turn(points: FloatArray, angle: float) -> FloatArray:
    cos, sin = math.cos(angle), math.sin(angle)
    x, y = points[..., 0], points[..., 1]
    return np.stack([x * cos - y * sin, x * sin + y * cos], axis=-1)


class ChaseRenderer(RaceRenderer):
    """The race renderer, with the follow camera swapped for the chosen chase style."""

    def __init__(self, track: Track, car_size: tuple[float, float], size: tuple[int, int]):
        super().__init__(track, car_size, size)
        self.style = 7
        self.ease = SLOW_EASE
        self.swing_time = SWING_TIME
        self._up: float | None = None
        self._up_time = 0.0
        self._quarter: float | None = None
        self._calm: float | None = None  # the calm style's quarter turn
        self._swing: tuple[float, float] | None = None  # (start time, quarter it left)

    def camera(self, snapshot: Snapshot) -> Camera:
        if self.mode is not CameraMode.FOLLOW or self.style == 0:
            return super().camera(snapshot)
        car = self.followed % len(snapshot.cars)
        if self.style == 4:
            aim = self._road_direction(snapshot, car)
        elif self.style == 5:
            aim = self._quarter_turn(self._road_direction(snapshot, car))
        else:
            aim = float(snapshot.cars.yaw[car])
        elapsed = snapshot.time - self._up_time
        if self.style == 6:
            up = self._calm_quarter(snapshot, car, fresh=not 0 <= elapsed <= 1.0)
        elif self.style == 1 or self._up is None or not 0 <= elapsed <= 1.0:
            up = aim
        else:
            turn = float(wrap_angle(aim - self._up))
            step = turn * (1 - math.exp(-elapsed / self.ease))
            excess = abs(turn) - CATCH_UP
            if self.style == 7 and excess > 0:
                step += math.copysign(excess, turn) * (1 - math.exp(-elapsed / CATCH_UP_TIME))
            up = self._up + step
        self._up, self._up_time = up, snapshot.time
        center = snapshot.cars.position[car]
        if self.style == 3:
            ahead = LOOK_AHEAD * self.size[1] / self._follow_scale
            center = center + ahead * np.array([math.cos(up), math.sin(up)])
        return TurnedCamera(
            center=(float(center[0]), float(center[1])),
            scale=self._follow_scale,
            size=self.size,
            up=up,
        )

    def _road_direction(self, snapshot: Snapshot, car: int) -> float:
        """The way the road runs `ROAD_AHEAD` metres past the car: steering doesn't move it."""
        centerline = self.track.centerline
        samples = len(centerline.points)
        ahead = round(ROAD_AHEAD / (self.track.length / samples))
        tangent = centerline.tangent[(int(snapshot.race.segment[car]) + ahead) % samples]
        return math.atan2(float(tangent[1]), float(tangent[0]))

    def _quarter_turn(self, direction: float) -> float:
        """The nearest of north, east, south, west to ``direction``, kept until the road points
        `QUARTER_HOLD` away from it: the view stays still between a few big turns."""
        if (
            self._quarter is None
            or abs(float(wrap_angle(direction - self._quarter))) > QUARTER_HOLD
        ):
            self._quarter = round(direction / (math.pi / 2)) * (math.pi / 2)
        return self._quarter

    def _calm_quarter(self, snapshot: Snapshot, car: int, fresh: bool) -> float:
        """Quarter turns that wait for a straight (or `MUST_SWING`) and then swing smoothly."""
        road = self._road_direction(snapshot, car)
        now = snapshot.time
        if self._calm is None or fresh:
            self._calm, self._swing = round(road / (math.pi / 2)) * (math.pi / 2), None
        if self._swing is not None:
            start, before = self._swing
            progress = (now - start) / self.swing_time
            if progress < 1:
                eased = progress * progress * (3 - 2 * progress)  # smoothstep: gentle both ends
                return before + float(wrap_angle(self._calm - before)) * eased
            self._swing = None
        off = abs(float(wrap_angle(road - self._calm)))
        if off > QUARTER_HOLD and (off > MUST_SWING or self._straight_ahead(snapshot, car)):
            self._swing = (now, self._calm)
            self._calm = round(road / (math.pi / 2)) * (math.pi / 2)
            return self._swing[1]
        return self._calm

    def _straight_ahead(self, snapshot: Snapshot, car: int) -> bool:
        """Whether the next `STRAIGHT_AHEAD` seconds of road (20 m at least) barely bend."""
        centerline = self.track.centerline
        samples = len(centerline.points)
        reach = max(20.0, float(snapshot.cars.speed[car]) * STRAIGHT_AHEAD)
        count = max(2, round(reach / (self.track.length / samples)))
        indices = (int(snapshot.race.segment[car]) + np.arange(count)) % samples
        angles = np.arctan2(centerline.tangent[indices, 1], centerline.tangent[indices, 0])
        return bool(np.all(np.abs(wrap_angle(angles - angles[0])) < STRAIGHT_BEND))

    def draw(
        self,
        surface: pygame.Surface,
        snapshot: Snapshot,
        rays: FloatArray | None = None,
        hint: str = "",
    ) -> None:
        time = self.swing_time if self.style == 6 else self.ease
        hint = f"V style: {STYLES[self.style]}  ·  [ ] turn time {time:.2f} s"
        super().draw(surface, snapshot, rays, hint)


class PrototypeWindow(DriveWindow):
    def __init__(self, track: Track, config: RacecarConfig, title: str) -> None:
        super().__init__(track, config, title)
        car_size = (config.vehicle.length, config.vehicle.width)
        self.renderer = ChaseRenderer(track, car_size, self.renderer.size)

    def _press(self, key: int) -> None:
        renderer = self.renderer
        assert isinstance(renderer, ChaseRenderer)
        if key == pygame.K_v:
            renderer.style = (renderer.style + 1) % len(STYLES)
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            factor = 1.25 if key == pygame.K_RIGHTBRACKET else 1 / 1.25
            if renderer.style == 6:
                renderer.swing_time = float(np.clip(renderer.swing_time * factor, 0.2, 5.0))
            else:
                renderer.ease = float(np.clip(renderer.ease * factor, 0.05, 5.0))
        else:
            super()._press(key)


def main(arguments: list[str]) -> None:
    track_file = read_track_file(Path(arguments[0]))
    config = load_config([Path(name) for name in arguments[1:]])
    PrototypeWindow(
        track_file.to_track(), config, f"{track_file.name} (chase camera prototype)"
    ).run()


if __name__ == "__main__":
    main(sys.argv[1:])
