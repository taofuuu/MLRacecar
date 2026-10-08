"""Tests for mlracecar.play.replay: watching a replay, run off-screen (see tests/conftest.py)."""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pygame
import pytest

from mlracecar.config.models import RacecarConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.env.racing import RacingEnv
from mlracecar.io.replay import Replay
from mlracecar.io.track_file import TrackFile
from mlracecar.play.replay import SPEEDS, Playback, ReplayWindow, run_replay
from mlracecar.render.race import CameraMode, Overlay
from mlracecar.render.timeline import bar_rect
from snapshots import assert_same_snapshots

ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CIRCLE = TrackFile.from_arrays(
    "Circle", 60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 48
)
CONFIG = RacecarConfig()


def race(seconds: float) -> list[Snapshot]:
    """A car's snapshots, from the grid, at full throttle with a little left steering."""
    env = RacingEnv(CIRCLE.to_track(), CONFIG)
    env.reset(seed=0)
    assert env.world is not None
    snapshots = [env.world.snapshot]
    for _ in range(round(seconds / 0.05)):
        env.step(np.array([0.3, 0.6], dtype=np.float32))
        snapshots.append(env.world.snapshot)
    return snapshots


SNAPSHOTS = race(4.0)  # 81 snapshots, 0.05 s apart


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #


def test_a_replay_starts_at_the_beginning_playing_at_normal_speed() -> None:
    playback = Playback(SNAPSHOTS)

    assert playback.position == 0.0
    assert playback.duration == pytest.approx(4.0)
    assert playback.playing
    assert playback.speed == 1.0


def test_time_passes_at_the_playback_speed_and_stops_at_the_end() -> None:
    playback = Playback(SNAPSHOTS)

    playback.advance(0.5)
    assert playback.position == pytest.approx(0.5)
    playback.faster()
    playback.advance(0.5)  # x2
    assert playback.position == pytest.approx(1.5)
    playback.advance(10.0)
    assert playback.position == pytest.approx(4.0)
    assert not playback.playing  # stopped at the end


def test_paused_time_stands_still_and_playing_from_the_end_starts_again() -> None:
    playback = Playback(SNAPSHOTS)
    playback.toggle()

    playback.advance(1.0)
    assert (playback.position, playback.playing) == (0.0, False)
    playback.seek(99.0)
    playback.toggle()
    assert (playback.position, playback.playing) == (0.0, True)


def test_speeds_go_from_a_quarter_to_four_times() -> None:
    playback = Playback(SNAPSHOTS)

    for _ in range(9):
        playback.faster()
    assert playback.speed == 4.0
    for _ in range(9):
        playback.slower()
    assert playback.speed == 0.25
    assert SPEEDS == (0.25, 0.5, 1.0, 2.0, 4.0)


def test_seeking_stays_within_the_race() -> None:
    playback = Playback(SNAPSHOTS)

    playback.seek(-3.0)
    assert playback.position == 0.0
    playback.seek(5.0)
    assert playback.position == pytest.approx(4.0)


def test_at_a_snapshot_it_shows_that_snapshot_exactly() -> None:
    playback = Playback(SNAPSHOTS)

    for step in (0, 1, 40, 80):
        playback.seek(SNAPSHOTS[step].time)
        assert_same_snapshots([playback.snapshot()], [SNAPSHOTS[step]])


def test_between_snapshots_the_car_is_blended_towards_the_next() -> None:
    playback = Playback(SNAPSHOTS)
    before, after = SNAPSHOTS[20], SNAPSHOTS[21]

    playback.seek((before.time + after.time) / 2)
    shown = playback.snapshot()

    halfway = (before.cars.position + after.cars.position) / 2
    np.testing.assert_allclose(shown.cars.position, halfway)
    assert shown.time == pytest.approx((before.time + after.time) / 2)
    assert shown.race.distance[0] == after.race.distance[0]  # everything else is the next one's


def test_the_label_says_where_it_is_how_fast_and_whether_paused() -> None:
    playback = Playback(SNAPSHOTS)
    playback.seek(1.25)

    assert playback.label() == "0:01.250 / 0:04.000  x1"
    playback.slower()
    playback.toggle()
    assert playback.label() == "0:01.250 / 0:04.000  x0.5  paused"


def test_a_replay_of_one_snapshot_just_shows_it() -> None:
    playback = Playback(SNAPSHOTS[:1])

    playback.advance(1.0)

    assert (playback.duration, playback.position, playback.playing) == (0.0, 0.0, False)
    assert playback.snapshot() is SNAPSHOTS[0]


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #


def replay_of(snapshots: list[Snapshot]) -> Replay:
    return Replay(snapshots, CIRCLE, CONFIG.model_dump(mode="json"), {})


@pytest.fixture
def window() -> Iterator[ReplayWindow]:
    window = ReplayWindow(replay_of(SNAPSHOTS), CONFIG, "Circle - A, run 1")
    yield window
    pygame.display.quit()


def key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=0)


def click(kind: int, pos: tuple[int, int], **extra: Any) -> pygame.event.Event:
    return pygame.event.Event(kind, pos=pos, button=1, **extra)


def test_keys_play_pause_skip_and_change_the_speed(window: ReplayWindow) -> None:
    playback = window.playback

    window.handle(key(pygame.K_SPACE))
    assert not playback.playing
    window.handle(key(pygame.K_RIGHT))
    window.handle(key(pygame.K_RIGHT))
    window.handle(key(pygame.K_LEFT))
    assert playback.position == pytest.approx(1.0)
    window.handle(key(pygame.K_UP))
    assert playback.speed == 2.0
    window.handle(key(pygame.K_DOWN))
    window.handle(key(pygame.K_DOWN))
    assert playback.speed == 0.5
    window.handle(key(pygame.K_END))
    assert playback.position == pytest.approx(4.0)
    window.handle(key(pygame.K_HOME))
    assert playback.position == 0.0


def test_the_camera_and_overlays_change_as_when_driving(window: ReplayWindow) -> None:
    window.handle(key(pygame.K_c))
    window.handle(key(pygame.K_4))

    assert window.renderer.mode is not CameraMode.FOLLOW
    assert Overlay.RAYS in window.renderer.overlays
    window.draw()  # with the rays


def test_clicking_and_dragging_the_bar_goes_anywhere(window: ReplayWindow) -> None:
    bar = bar_rect(window._screen.get_size())

    window.handle(click(pygame.MOUSEBUTTONDOWN, (bar.left + bar.width // 4, bar.centery)))
    assert window.playback.position == pytest.approx(1.0, abs=0.02)
    window.handle(
        click(pygame.MOUSEMOTION, (bar.right, bar.centery), rel=(1, 0), buttons=(1, 0, 0))
    )
    assert window.playback.position == pytest.approx(4.0)
    window.handle(click(pygame.MOUSEBUTTONUP, (bar.right, bar.centery)))
    window.handle(click(pygame.MOUSEMOTION, (bar.left, bar.centery), rel=(1, 0), buttons=(0, 0, 0)))
    assert window.playback.position == pytest.approx(4.0)  # let go: no more scrubbing


def test_dragging_elsewhere_moves_the_view(window: ReplayWindow) -> None:
    window.handle(click(pygame.MOUSEBUTTONDOWN, (100, 100)))
    window.handle(click(pygame.MOUSEMOTION, (140, 100), rel=(40, 0), buttons=(1, 0, 0)))

    assert window.renderer.mode is CameraMode.FREE
    assert window.playback.position == 0.0


def test_the_wheel_zooms(window: ReplayWindow) -> None:
    snapshot = window.playback.snapshot()
    scale = window.renderer.camera(snapshot).scale

    window.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=2))

    assert window.renderer.camera(snapshot).scale == pytest.approx(scale * 1.15**2)


def test_the_window_can_be_resized(window: ReplayWindow) -> None:
    pygame.display.set_mode((400, 300))  # what the system does when the window is resized
    window.handle(pygame.event.Event(pygame.VIDEORESIZE, size=(400, 300), w=400, h=300))
    window.draw()

    assert window.renderer.size == (400, 300)


def test_the_window_shows_the_race_and_the_timeline(window: ReplayWindow) -> None:
    window.playback.seek(2.0)
    window.draw()

    assert pygame.display.get_caption()[0] == "Circle - A, run 1 - Replay"
    surface = pygame.display.get_surface()
    assert surface is not None
    bar = bar_rect(surface.get_size())
    played = surface.get_at((bar.left + bar.width // 4, bar.centery))[:3]
    unplayed = surface.get_at((bar.right - bar.width // 4, bar.centery))[:3]
    assert played == (96, 165, 250)  # the played half of the bar
    assert played != unplayed


def test_other_keys_change_nothing(window: ReplayWindow) -> None:
    window.handle(key(pygame.K_q))

    assert (window.playback.position, window.playback.playing) == (0.0, True)
    assert window.renderer.overlays == set()


def test_escape_or_closing_ends_it(window: ReplayWindow) -> None:
    window.handle(key(pygame.K_ESCAPE))
    assert not window.running
    window.running = True
    window.handle(pygame.event.Event(pygame.QUIT))
    assert not window.running


def test_run_replay_plays_until_closed() -> None:
    pygame.display.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    run_replay(replay_of(SNAPSHOTS), CONFIG, "Circle")  # returns once the window is closed
