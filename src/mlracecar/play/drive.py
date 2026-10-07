"""The driving window behind `racecar drive`: you drive a car round a track with the keyboard.

The race runs in real time, one driver decision every `Timing.decision_dt` (0.05 s by default),
while the window draws 60 frames a second and blends the car between decisions so it moves
smoothly. The `KeyboardAgent` turns the keys held down into steering and a pedal.
"""

import math
from collections.abc import Callable

import numpy as np
import pygame

from mlracecar.agents.keyboard import HeldKeys, KeyboardAgent
from mlracecar.config.models import RacecarConfig
from mlracecar.core.race.events import LapCompleted
from mlracecar.core.sensors import RaySensor
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.render.hud import lap_time_text
from mlracecar.render.race import Overlay, RaceRenderer, interpolated

WINDOW_SIZE = (1280, 800)
"""The window's size, unless the screen is too small for it."""

SCREEN_SHARE = 0.85
"""At most this share of the screen's width and height goes to the window."""

FRAME_RATE = 60

MAX_CATCH_UP = 0.25
"""Seconds of race the window will run at once to catch up after a stall (dragging the window,
say); any more is skipped rather than played in a burst."""

LEFT = (pygame.K_LEFT, pygame.K_a)
RIGHT = (pygame.K_RIGHT, pygame.K_d)
THROTTLE = (pygame.K_UP, pygame.K_w)
BRAKE = (pygame.K_DOWN, pygame.K_s)

OVERLAY_KEYS = {
    pygame.K_1: Overlay.CENTERLINE,
    pygame.K_2: Overlay.CHECKPOINTS,
    pygame.K_3: Overlay.VELOCITY,
    pygame.K_4: Overlay.RAYS,
}

HINT = "Arrows or WASD: drive  ·  R: restart  ·  C: camera  ·  1-4: overlays  ·  P: pause"

_NO_OBSERVATIONS = np.zeros((1, 0), dtype=np.float32)
"""The keyboard agent needs no observations: the person sees the screen."""


def held_keys(pressed: Callable[[int], bool]) -> HeldKeys:
    """The driving keys held down, from a lookup of whether each key is pressed."""
    return HeldKeys(
        left=any(pressed(key) for key in LEFT),
        right=any(pressed(key) for key in RIGHT),
        throttle=any(pressed(key) for key in THROTTLE),
        brake=any(pressed(key) for key in BRAKE),
    )


class DriveWindow:
    """A window in which one car is driven round a track with the keyboard.

    Args:
        track: The track.
        config: The settings: the car, the timing, the race rules, and the sensors.
        title: Shown in the window's title bar, such as the track's name.
        seed: Seeds the world's randomness.
    """

    def __init__(self, track: Track, config: RacecarConfig, title: str, seed: int = 0) -> None:
        pygame.display.init()
        self._screen = pygame.display.set_mode(_fitting_window_size(), pygame.RESIZABLE)
        car = config.vehicle.to_params()
        self.timing = config.simulation.to_timing()
        self.world = World(
            track,
            KinematicBicycle(car),
            self.timing,
            1,
            np.random.default_rng(seed),
            settings=config.race.to_settings(),
        )
        self.agent = KeyboardAgent(self.timing.decision_dt)
        self.sensor = RaySensor(track, config.sensors.to_settings())
        """The distance rays the AI will drive by; key 4 shows them."""
        self.renderer = RaceRenderer(track, (car.length, car.width), self._screen.get_size())
        self.title = title
        self.running = True
        self.paused = False
        self.session_best = math.nan
        """The best valid lap since the window opened, restarts included."""
        self._previous = self.world.snapshot
        self._owed = 0.0  # seconds of race not yet run
        self._dragging = False

    def handle(self, event: pygame.event.Event) -> None:
        """React to one pygame event (keys, mouse, window)."""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self._screen = pygame.display.get_surface() or self._screen
            self.renderer.resize(self._screen.get_size())
        elif event.type == pygame.MOUSEWHEEL:
            self.renderer.zoom_at(pygame.mouse.get_pos(), 1.15**event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            self.renderer.pan(*event.rel)
        elif event.type == pygame.KEYDOWN:
            self._press(event.key)

    def _press(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_r:
            self.restart()
        elif key == pygame.K_c:
            self.renderer.next_camera()
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key in OVERLAY_KEYS:
            self.renderer.toggle(OVERLAY_KEYS[key])

    def advance(self, seconds: float, keys: HeldKeys) -> None:
        """Let ``seconds`` of real time pass, with ``keys`` held: run each decision that's due."""
        if self.paused:
            return
        self._owed = min(self._owed + seconds, MAX_CATCH_UP)
        while self._owed >= self.timing.decision_dt:
            self._owed -= self.timing.decision_dt
            self.agent.keys = keys
            self._previous = self.world.snapshot
            snapshot = self.world.step(self.agent.act(_NO_OBSERVATIONS))
            laps = [
                event.time
                for event in snapshot.events
                if isinstance(event, LapCompleted) and event.valid
            ]
            self.session_best = min([self.session_best, *laps], key=_slowest_if_none)

    def restart(self) -> None:
        """Put the car back on the grid, at rest, with its race starting over."""
        self.agent.reset()
        self._previous = self.world.reset()
        self._owed = 0.0

    def draw(self) -> None:
        """Draw the race, the car blended between its last two decisions, and show it."""
        fraction = self._owed / self.timing.decision_dt
        snapshot = interpolated(self._previous, self.world.snapshot, fraction)
        rays = self.sensor.sense(snapshot).end if Overlay.RAYS in self.renderer.overlays else None
        self.renderer.draw(self._screen, snapshot, rays, hint=HINT)
        pygame.display.flip()
        best = lap_time_text(self.session_best)
        paused = " - paused" if self.paused else ""
        pygame.display.set_caption(f"{self.title} - best this session {best}{paused} - Drive")

    def run(self) -> None:
        """Run until the window is closed or Esc is pressed."""
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle(event)
            keys = held_keys(pygame.key.get_pressed().__getitem__)
            self.advance(clock.tick(FRAME_RATE) / 1000, keys)
            self.draw()
        pygame.display.quit()


def run_drive(track: Track, config: RacecarConfig, title: str) -> None:
    """Open the driving window and run it until it's closed."""
    DriveWindow(track, config, title).run()


def _slowest_if_none(seconds: float) -> float:
    """Sort key for lap times: NaN (no time yet) is slower than any lap."""
    return math.inf if math.isnan(seconds) else seconds


def _fitting_window_size() -> tuple[int, int]:
    """`WINDOW_SIZE`, shrunk to fit on smaller screens."""
    screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
    return (
        min(WINDOW_SIZE[0], int(screen_width * SCREEN_SHARE)),
        min(WINDOW_SIZE[1], int(screen_height * SCREEN_SHARE)),
    )
