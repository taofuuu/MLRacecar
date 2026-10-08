"""The replay window behind `racecar replay`: watch a recorded race, at any speed, and scrub;
or save it as a video (`export_replay`, ``racecar replay --export``).

`Playback` is where the replay is up to: the time, the speed, and whether it's playing. It's
plain logic, tested without a window. `ReplayWindow` draws the race as the driving window does,
60 frames a second with the cars blended between snapshots, and the timeline along the bottom.
A video is drawn the same way, offscreen, a picture every 1/fps seconds of the race.
"""

import math
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pygame

from mlracecar.config.models import RacecarConfig
from mlracecar.core.race.events import LapCompleted
from mlracecar.core.sensors import RaySensor
from mlracecar.core.snapshot import Snapshot
from mlracecar.io.replay import Replay
from mlracecar.play.drive import FRAME_RATE, OVERLAY_KEYS, fitting_window_size
from mlracecar.render.hud import lap_time_text
from mlracecar.render.race import CameraMode, Overlay, RaceRenderer, interpolated
from mlracecar.render.timeline import Timeline, fraction_at, grabs
from mlracecar.render.video import Frame, VideoError, write_video

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
"""The playback speeds, slowest first."""

SKIP = 1.0
"""Seconds that Left and Right jump back or on."""

VIDEO_SIZE = (960, 600)
"""A video's width and height in pixels, unless asked otherwise."""

VIDEO_FPS = 25
"""A video's pictures a second, unless asked otherwise: a GIF plays 25 exactly (40 ms each)."""

HINT = "Space: play/pause  ·  Left/Right: 1 s  ·  Up/Down: speed  ·  C: camera  ·  1-4: overlays"


class Playback:
    """Where a replay is up to.

    Args:
        snapshots: The race, in order, evenly spaced in time.
    """

    def __init__(self, snapshots: Sequence[Snapshot]) -> None:
        self.snapshots = snapshots
        self._times = np.array([snapshot.time for snapshot in snapshots]) - snapshots[0].time
        self.duration = float(self._times[-1])
        """Seconds from the first snapshot to the last."""
        self.position = 0.0
        """Seconds from the start."""
        self.playing = True
        self._speed = SPEEDS.index(1.0)

    @property
    def speed(self) -> float:
        """Seconds of race per second of watching."""
        return SPEEDS[self._speed]

    @property
    def at_end(self) -> bool:
        return self.position >= self.duration

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` of real time pass. Playing stops at the end."""
        if self.playing:
            self.seek(self.position + seconds * self.speed)
            if self.at_end:
                self.playing = False

    def seek(self, position: float) -> None:
        """Go to ``position`` seconds from the start (kept within the race)."""
        self.position = min(max(position, 0.0), self.duration)

    def toggle(self) -> None:
        """Play or pause. Playing from the end starts again from the beginning."""
        if not self.playing and self.at_end:
            self.position = 0.0
        self.playing = not self.playing

    def faster(self) -> None:
        """The next speed up, up to the fastest."""
        self._speed = min(self._speed + 1, len(SPEEDS) - 1)

    def slower(self) -> None:
        """The next speed down, down to the slowest."""
        self._speed = max(self._speed - 1, 0)

    def snapshot(self) -> Snapshot:
        """The race at the current position: the snapshot there, or between two, the cars
        blended from the one before towards the one after (as the driving window shows them)."""
        after = int(np.searchsorted(self._times, self.position))  # the first at or after it
        if after == 0 or self._times[after] == self.position:
            return self.snapshots[after]
        before = after - 1
        span = self._times[after] - self._times[before]
        fraction = (self.position - self._times[before]) / span
        return interpolated(self.snapshots[before], self.snapshots[after], float(fraction))

    def label(self) -> str:
        """Where it's up to, as text: ``0:12.350 / 1:00.000  x1  paused``."""
        speed = f"x{self.speed:g}"
        state = "" if self.playing else "  paused"
        return f"{lap_time_text(self.position)} / {lap_time_text(self.duration)}  {speed}{state}"


class ReplayWindow:
    """A window that plays a replay.

    Args:
        replay: The recorded race.
        config: Its settings (from the replay), for the car's size and the distance rays.
        title: Shown in the window's title bar.
    """

    def __init__(
        self,
        replay: Replay,
        config: RacecarConfig,
        title: str,
        camera: CameraMode = CameraMode.FOLLOW,
        rays: bool = False,
    ) -> None:
        pygame.display.init()
        self._screen = pygame.display.set_mode(fitting_window_size(), pygame.RESIZABLE)
        track = replay.track.to_track()
        car = config.vehicle.to_params()
        self.playback = Playback(replay.snapshots)
        self.sensor = RaySensor(track, config.sensors.to_settings())
        """The distance rays the car drove by; key 4 shows them."""
        self.renderer = RaceRenderer(track, (car.length, car.width), self._screen.get_size())
        self.renderer.mode = camera
        if rays:
            self.renderer.overlays.add(Overlay.RAYS)
        self.timeline = Timeline()
        self.title = title
        self.running = True
        self._scrubbing = False
        self._panning = False

    def handle(self, event: pygame.event.Event) -> None:
        """React to one pygame event (keys, mouse, window)."""
        size = self._screen.get_size()
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self._screen = pygame.display.get_surface() or self._screen
            self.renderer.resize(self._screen.get_size())
        elif event.type == pygame.MOUSEWHEEL:
            self.renderer.zoom_at(pygame.mouse.get_pos(), 1.15**event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if grabs(size, event.pos):
                self._scrubbing = True
                self._scrub(event.pos[0])
            else:
                self._panning = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._scrubbing = self._panning = False
        elif event.type == pygame.MOUSEMOTION:
            if self._scrubbing:
                self._scrub(event.pos[0])
            elif self._panning:
                self.renderer.pan(*event.rel)
        elif event.type == pygame.KEYDOWN:
            self._press(event.key)

    def _scrub(self, x: float) -> None:
        playback = self.playback
        playback.seek(fraction_at(self._screen.get_size(), x) * playback.duration)

    def _press(self, key: int) -> None:
        playback = self.playback
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            playback.toggle()
        elif key == pygame.K_LEFT:
            playback.seek(playback.position - SKIP)
        elif key == pygame.K_RIGHT:
            playback.seek(playback.position + SKIP)
        elif key == pygame.K_UP:
            playback.faster()
        elif key == pygame.K_DOWN:
            playback.slower()
        elif key == pygame.K_HOME:
            playback.seek(0.0)
        elif key == pygame.K_END:
            playback.seek(playback.duration)
        elif key == pygame.K_c:
            self.renderer.next_camera()
        elif key in OVERLAY_KEYS:
            self.renderer.toggle(OVERLAY_KEYS[key])

    def draw(self) -> None:
        """Draw the race where the replay is up to, and the timeline, and show it."""
        playback = self.playback
        snapshot = playback.snapshot()
        rays = self.sensor.sense(snapshot).end if Overlay.RAYS in self.renderer.overlays else None
        self.renderer.draw(self._screen, snapshot, rays, hint=HINT)
        fraction = playback.position / playback.duration if playback.duration else 1.0
        self.timeline.draw(self._screen, fraction, playback.label())
        pygame.display.flip()
        pygame.display.set_caption(f"{self.title} - Replay")

    def run(self) -> None:
        """Run until the window is closed or Esc is pressed."""
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                self.handle(event)
            self.playback.advance(clock.tick(FRAME_RATE) / 1000)
            self.draw()
        pygame.display.quit()


def run_replay(
    replay: Replay,
    config: RacecarConfig,
    title: str,
    camera: CameraMode = CameraMode.FOLLOW,
    rays: bool = False,
    start: float = 0.0,
) -> None:
    """Open the replay window, ``start`` seconds in, and run it until it's closed."""
    window = ReplayWindow(replay, config, title, camera, rays)
    window.playback.seek(start)
    window.run()


def lap_times(replay: Replay, lap: int) -> tuple[float, float]:
    """When the replay's ``lap``-th valid lap (counting from 1) starts and ends, in seconds
    from the start of the replay.

    Raises:
        VideoError: If the replay has fewer valid laps.
    """
    laps = [
        event
        for snapshot in replay.snapshots
        for event in snapshot.events
        if isinstance(event, LapCompleted) and event.valid
    ]
    if not 1 <= lap <= len(laps):
        valid = f"{len(laps)} valid lap{'' if len(laps) == 1 else 's'}"
        raise VideoError(f"there's no lap {lap}: the replay has {valid}")
    begin = replay.snapshots[0].time
    end = laps[lap - 1].at - begin
    return end - laps[lap - 1].time, end


def replay_frames(
    replay: Replay,
    config: RacecarConfig,
    *,
    camera: CameraMode = CameraMode.FOLLOW,
    rays: bool = False,
    start: float = 0.0,
    end: float | None = None,
    fps: float = VIDEO_FPS,
    size: tuple[int, int] = VIDEO_SIZE,
) -> Iterator[Frame]:
    """The replay as pictures, drawn offscreen as the window draws it (without the bottom line),
    one every ``1 / fps`` seconds of the race from ``start`` to ``end`` (the end by default).

    Raises:
        VideoError: If the times aren't within the replay, in order, or the rate or size isn't
            positive.
    """
    playback = Playback(replay.snapshots)
    end = playback.duration if end is None else end
    if not 0 <= start < end <= playback.duration + 1e-9:
        raise VideoError(
            f"--from and --to must be in order, within the replay's {playback.duration:g} s; "
            f"got {start:g} to {end:g}"
        )
    if fps <= 0 or min(size) < 1:
        raise VideoError(f"the frame rate and size must be positive, got {fps:g} and {size}")
    track = replay.track.to_track()
    car = config.vehicle.to_params()
    renderer = RaceRenderer(track, (car.length, car.width), size)
    renderer.mode = camera
    renderer.caption = False
    sensor = RaySensor(track, config.sensors.to_settings()) if rays else None
    if rays:
        renderer.overlays.add(Overlay.RAYS)
    count = math.floor((end - start) * fps + 1e-9) + 1

    def frames() -> Iterator[Frame]:
        for index in range(count):
            playback.seek(start + index / fps)
            snapshot = playback.snapshot()
            yield renderer.render(snapshot, None if sensor is None else sensor.sense(snapshot).end)

    return frames()


def export_replay(
    replay: Replay,
    config: RacecarConfig,
    path: str | Path,
    *,
    camera: CameraMode = CameraMode.FOLLOW,
    rays: bool = False,
    start: float = 0.0,
    end: float | None = None,
    fps: float = VIDEO_FPS,
    size: tuple[int, int] = VIDEO_SIZE,
) -> None:
    """Save the replay, from ``start`` to ``end`` seconds, as a video: a GIF or an MP4, by the
    extension of ``path`` (`write_video`).

    Raises:
        VideoError: As for `replay_frames` and `write_video`.
    """
    frames = replay_frames(
        replay, config, camera=camera, rays=rays, start=start, end=end, fps=fps, size=size
    )
    write_video(path, frames, fps)
