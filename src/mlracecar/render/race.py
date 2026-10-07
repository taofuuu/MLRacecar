"""Drawing a race: the track with kerbs, the cars, the HUD, and debug overlays.

`RaceRenderer` draws from snapshots alone, so the same code serves live play, replays, and
video. It never runs or changes the simulation (an import-linter contract checks that). It can
draw into a window or, offscreen, return each frame as an RGB array.
"""

import math
from dataclasses import replace
from enum import StrEnum

import numpy as np
import pygame
from numpy.typing import NDArray

from mlracecar.core.geometry import FloatArray, rotate, wrap_angle
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.render.camera import MAX_SCALE, MIN_SCALE, Camera
from mlracecar.render.drawing import draw_track
from mlracecar.render.hud import Hud

type Color = tuple[int, int, int]

GRASS: Color = (34, 58, 38)
KERB_RED: Color = (200, 40, 40)
KERB_WHITE: Color = (235, 235, 235)
CENTERLINE: Color = (94, 234, 212)
CHECKPOINT: Color = (100, 116, 139)
NEXT_CHECKPOINT: Color = (74, 222, 128)
VELOCITY: Color = (56, 189, 248)
RAY: Color = (232, 121, 249)
CAR_OUTLINE: Color = (15, 17, 20)
FOLLOWED_OUTLINE: Color = (250, 204, 21)
WINDSCREEN: Color = (30, 41, 59)
CAR_COLORS: tuple[Color, ...] = (
    (59, 130, 246),
    (239, 68, 68),
    (34, 197, 94),
    (249, 115, 22),
    (168, 85, 247),
    (236, 72, 153),
    (20, 184, 166),
    (234, 179, 8),
)
"""Car colours, in turn; car 0 is blue."""

FOLLOW_SCALE = 6.0
"""Starting zoom of the follow camera, in pixels per metre."""

VIEW_AHEAD = 2.5
"""Seconds of road ahead of the followed car that the follow camera keeps on screen: it
looks further ahead, and zooms out if it must, as the car speeds up."""

CAMERA_LAG = 0.5
"""Seconds the follow camera takes to make most of a move (63%): it eases towards where it
should be instead of jumping when the car brakes hard or the steering flicks."""

EASE_GAP = 1.0
"""Seconds without a frame after which the follow camera jumps straight to its place."""

FOLLOW_MARGIN = 40
"""Pixels the follow camera keeps between the road ahead and the edge of the window."""

HUD_SPACE = 180
"""Pixels on the left that the overview keeps clear of the track, for the HUD."""

KERB_RADIUS = 100.0
"""Bends tighter than this many metres get kerbs on both edges."""
KERB_WIDTH = 1.0
"""Metres."""
KERB_BLOCK = 3.0
"""Length of each red or white block, in metres."""

VELOCITY_SECONDS = 0.5
"""Velocity arrows show where each car will be in this many seconds."""


class CameraMode(StrEnum):
    """What the camera shows."""

    FOLLOW = "follow"
    """The followed car, in the middle of the window."""
    OVERVIEW = "overview"
    """The whole track."""
    FREE = "free"
    """Wherever it was panned and zoomed to."""


class Overlay(StrEnum):
    """Debug drawings that can be switched on and off."""

    CENTERLINE = "centerline"
    """The middle of the road, which progress is measured along."""
    CHECKPOINTS = "checkpoints"
    """Every checkpoint line, with the followed car's next one highlighted."""
    VELOCITY = "velocity"
    """An arrow from each car to where it will be in `VELOCITY_SECONDS`."""
    RAYS = "rays"
    """Sensor rays, when there are any (they arrive with the RL environment)."""


class RaceRenderer:
    """Draws a race on one track.

    Args:
        track: The track.
        car_size: Length and width of the cars, in metres.
        size: Window or image width and height, in pixels.
    """

    def __init__(
        self, track: Track, car_size: tuple[float, float], size: tuple[int, int] = (1280, 800)
    ) -> None:
        self.track = track
        self.car_size = car_size
        self.mode = CameraMode.FOLLOW
        self.followed = 0
        self.overlays: set[Overlay] = set()
        self._edges = np.concatenate([track.left, track.right])
        self._overview = _overview(self._edges, size)
        self._free = self._overview
        self._follow_scale = FOLLOW_SCALE
        self._lead: FloatArray | None = None  # the follow camera's offset ahead of the car
        self._lead_car = -1
        self._lead_time = 0.0
        self._kerbs, self._kerb_colors = _kerb_blocks(track)
        self._hud = Hud()
        self._offscreen: pygame.Surface | None = None

    @property
    def size(self) -> tuple[int, int]:
        """Width and height in pixels."""
        return self._overview.size

    def resize(self, size: tuple[int, int]) -> None:
        """Draw at a new size (after the window was resized, say)."""
        self._overview = _overview(self._edges, size)
        self._free = self._free.resized(size)
        self._offscreen = None

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #

    def camera(self, snapshot: Snapshot) -> Camera:
        """The camera for this snapshot, in the current mode."""
        if self.mode is CameraMode.OVERVIEW:
            return self._overview
        if self.mode is CameraMode.FREE:
            return self._free
        return self._follow_camera(snapshot, self.followed % len(snapshot.cars))

    def _follow_camera(self, snapshot: Snapshot, car: int) -> Camera:
        """About `VIEW_AHEAD` / 2 seconds ahead of the car, the way it points, zoomed out from the
        chosen zoom if that's needed to show `VIEW_AHEAD` seconds of road.

        The camera eases towards that place (`CAMERA_LAG`) rather than jumping there, and follows
        the car's heading rather than the exact direction it moves in, which flicks with every
        tap of the steering.
        """
        cars = snapshot.cars
        heading = float(cars.yaw[car])
        target = (
            float(cars.speed[car]) * VIEW_AHEAD / 2 * np.array([np.cos(heading), np.sin(heading)])
        )
        lead = self._ease(target, car, snapshot.time)
        center = cars.position[car] + lead
        ahead = float(np.hypot(lead[0], lead[1]))
        scale = self._follow_scale
        if ahead > 0:
            scale = min(scale, (min(self.size) / 2 - FOLLOW_MARGIN) / ahead)
        return Camera(center=(float(center[0]), float(center[1])), scale=scale, size=self.size)

    def _ease(self, target: FloatArray, car: int, time: float) -> FloatArray:
        """The follow camera's offset ahead of the car, moved part of the way towards
        ``target`` for the time since the last frame; all the way for a new car, or after a
        pause in the frames or a jump back in time."""
        elapsed = time - self._lead_time
        if self._lead is None or car != self._lead_car or not 0 <= elapsed <= EASE_GAP:
            lead = target
        else:
            lead = self._lead + (target - self._lead) * (1 - math.exp(-elapsed / CAMERA_LAG))
        self._lead, self._lead_car, self._lead_time = lead, car, time
        return lead

    def next_camera(self) -> CameraMode:
        """Switch to the next camera mode: follow, overview, free, and round again."""
        modes = list(CameraMode)
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        return self.mode

    def zoom_at(self, pixel: tuple[float, float], factor: float) -> None:
        """Zoom in (``factor`` > 1) or out. The overview camera becomes the free camera."""
        if self.mode is CameraMode.FOLLOW:
            self._follow_scale = float(np.clip(self._follow_scale * factor, MIN_SCALE, MAX_SCALE))
            return
        if self.mode is CameraMode.OVERVIEW:
            self._free, self.mode = self._overview, CameraMode.FREE
        self._free = self._free.zoom_at(pixel, factor)

    def pan(self, dx: float, dy: float) -> None:
        """Move the view by pixels; switches to the free camera, starting from what was shown."""
        if self.mode is CameraMode.OVERVIEW:
            self._free = self._overview
        self.mode = CameraMode.FREE
        self._free = self._free.pan(dx, dy)

    def toggle(self, overlay: Overlay) -> bool:
        """Switch a debug overlay on or off; returns whether it is now on."""
        self.overlays ^= {overlay}
        return overlay in self.overlays

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def draw(
        self,
        surface: pygame.Surface,
        snapshot: Snapshot,
        rays: FloatArray | None = None,
        hint: str = "",
    ) -> None:
        """Draw a snapshot onto a surface of `size`.

        Args:
            surface: Where to draw, such as the window.
            snapshot: The race at one moment.
            rays: Optional sensor ray end points in metres, shape ``(N, R, 2)``.
            hint: Extra text for the bottom line, such as the keys to press.
        """
        camera = self.camera(snapshot)
        followed = self.followed % len(snapshot.cars)
        surface.fill(GRASS)
        draw_track(surface, self.track, camera)
        self._draw_kerbs(surface, camera)
        if Overlay.CENTERLINE in self.overlays:
            points = camera.to_screen(self.track.centerline.points)
            pygame.draw.aalines(surface, CENTERLINE, True, points)
        if Overlay.CHECKPOINTS in self.overlays:
            self._draw_checkpoints(surface, camera, int(snapshot.race.checkpoint[followed]))
        if Overlay.RAYS in self.overlays and rays is not None:
            _draw_rays(surface, camera, snapshot.cars.position, rays)
        self._draw_cars(surface, camera, snapshot.cars, followed)
        if Overlay.VELOCITY in self.overlays:
            _draw_velocity(surface, camera, snapshot.cars)
        caption = f"Car {followed + 1} of {len(snapshot.cars)}  ·  camera: {self.mode.value}"
        self._hud.draw(surface, snapshot, followed, f"{caption}  ·  {hint}" if hint else caption)

    def render(self, snapshot: Snapshot, rays: FloatArray | None = None) -> NDArray[np.uint8]:
        """Draw a snapshot offscreen and return it as an RGB image, shape ``(height, width, 3)``.

        Needs no window, so it works headless (with SDL's dummy video driver, in CI).
        """
        if self._offscreen is None:
            self._offscreen = pygame.Surface(self.size)
        self.draw(self._offscreen, snapshot, rays)
        width, height = self.size
        pixels = pygame.image.tobytes(self._offscreen, "RGB")  # row by row, much faster
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()

    def _draw_kerbs(self, surface: pygame.Surface, camera: Camera) -> None:
        if KERB_WIDTH * camera.scale < 1.5 or len(self._kerbs) == 0:
            return  # thinner than a pixel or two: not worth drawing
        screen = camera.to_screen(self._kerbs)
        low, high = screen.min(axis=1), screen.max(axis=1)
        visible = np.all((high >= 0) & (low <= camera.size), axis=1)
        for polygon, red in zip(screen[visible], self._kerb_colors[visible], strict=True):
            pygame.draw.polygon(surface, KERB_RED if red else KERB_WHITE, polygon)

    def _draw_checkpoints(self, surface: pygame.Surface, camera: Camera, last: int) -> None:
        checkpoints = self.track.checkpoints
        left, right = camera.to_screen(checkpoints.left), camera.to_screen(checkpoints.right)
        upcoming = (last + 1) % len(left)
        for index, (start, end) in enumerate(zip(left, right, strict=True)):
            color = NEXT_CHECKPOINT if index == upcoming else CHECKPOINT
            pygame.draw.line(surface, color, start, end, width=3 if index == upcoming else 1)

    def _draw_cars(
        self, surface: pygame.Surface, camera: Camera, cars: VehicleState, followed: int
    ) -> None:
        length, width = self.car_size
        if length * camera.scale < 4:  # too small to see a shape: draw dots
            for index, point in enumerate(camera.to_screen(cars.position)):
                color = CAR_COLORS[index % len(CAR_COLORS)]
                pygame.draw.circle(surface, color, point, 4 if index == followed else 3)
            return
        bodies = camera.to_screen(_car_outline(cars, length, width, front=0.5, back=-0.5))
        windscreens = camera.to_screen(_car_outline(cars, length, 0.8 * width, front=0.3, back=0.1))
        for index in range(len(cars)):
            pygame.draw.polygon(surface, CAR_COLORS[index % len(CAR_COLORS)], bodies[index])
            pygame.draw.polygon(surface, WINDSCREEN, windscreens[index])
            outline = FOLLOWED_OUTLINE if index == followed else CAR_OUTLINE
            pygame.draw.polygon(
                surface, outline, bodies[index], width=2 if index == followed else 1
            )


def interpolated(before: Snapshot, after: Snapshot, fraction: float) -> Snapshot:
    """The race ``fraction`` of the way from ``before`` to ``after``, for drawing.

    The simulation moves in steps (20 a second by default), but a window draws 60 frames a
    second; drawing the cars between two snapshots makes their motion smooth. Only the cars'
    positions, headings, and the clock are blended; everything else is ``after``'s.
    """
    old, new = before.cars, after.cars
    if len(old) != len(new):
        return after
    cars = replace(
        new,
        x=old.x + (new.x - old.x) * fraction,
        y=old.y + (new.y - old.y) * fraction,
        yaw=wrap_angle(old.yaw + wrap_angle(new.yaw - old.yaw) * fraction),
    )
    return replace(after, cars=cars, time=before.time + (after.time - before.time) * fraction)


def _overview(edges: FloatArray, size: tuple[int, int]) -> Camera:
    """A camera showing the whole track, clear of the HUD on the left where there's room."""
    width, height = size
    space = HUD_SPACE if width > 2 * HUD_SPACE else 0
    fitted = Camera(size=(width - space, height)).fit(edges)
    center = (fitted.center[0] - space / 2 / fitted.scale, fitted.center[1])
    return Camera(center=center, scale=fitted.scale, size=size)


def _car_outline(
    cars: VehicleState, length: float, width: float, *, front: float, back: float
) -> FloatArray:
    """A rectangle along each car, from ``back`` to ``front`` times its length ahead of its
    centre, in metres: shape ``(N, 4, 2)``."""
    corners = np.array(
        [[front * length, width / 2], [front * length, -width / 2],
         [back * length, -width / 2], [back * length, width / 2]]
    )  # fmt: skip
    turned: FloatArray = rotate(corners[None, :, :], cars.yaw[:, None])
    return turned + cars.position[:, None, :]


def _draw_velocity(surface: pygame.Surface, camera: Camera, cars: VehicleState) -> None:
    velocity = rotate(np.column_stack([cars.vx, cars.vy]), cars.yaw)
    starts = camera.to_screen(cars.position)
    ends = camera.to_screen(cars.position + velocity * VELOCITY_SECONDS)
    for start, end in zip(starts, ends, strict=True):
        pygame.draw.line(surface, VELOCITY, start, end, width=2)


def _draw_rays(
    surface: pygame.Surface, camera: Camera, positions: FloatArray, rays: FloatArray
) -> None:
    starts = camera.to_screen(positions)
    for start, ends in zip(starts, camera.to_screen(rays), strict=True):
        for end in ends:
            pygame.draw.aaline(surface, RAY, start, end)


def _kerb_blocks(track: Track) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Red and white kerb blocks along both edges wherever the road bends tighter than
    `KERB_RADIUS`.

    Returns:
        Block outlines in metres, shape ``(B, P, 2)``, and whether each block is red.
    """
    line = track.centerline
    count = len(line.points)
    per_block = max(1, round(KERB_BLOCK / (line.length / count)))
    starts = np.arange(0, count - per_block + 1, per_block)
    samples = (starts[:, None] + np.arange(per_block + 1)) % count  # (B, per_block + 1)
    tight = np.abs(line.curvature) > 1 / KERB_RADIUS
    kerbed = tight[samples].any(axis=1)
    polygons = []
    colors = []
    for edge, inward in ((track.left, -line.normal), (track.right, line.normal)):
        inner = edge + inward * KERB_WIDTH
        for block, sample in zip(np.flatnonzero(kerbed), samples[kerbed], strict=True):
            polygons.append(np.concatenate([edge[sample], inner[sample[::-1]]]))
            colors.append(block % 2 == 0)
    if not polygons:
        return np.zeros((0, 2 * (per_block + 1), 2)), np.zeros(0, dtype=bool)
    return np.array(polygons), np.array(colors)
