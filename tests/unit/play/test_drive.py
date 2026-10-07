"""Tests for mlracecar.play.drive: the driving window, run off-screen (see tests/conftest.py)."""

import math
from collections.abc import Iterator

import numpy as np
import pygame
import pytest

from mlracecar.agents.keyboard import HeldKeys
from mlracecar.config.models import RacecarConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.play.drive import DriveWindow, held_keys, run_drive
from mlracecar.render.race import CameraMode, Overlay

ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 12)
DT = 0.05


@pytest.fixture
def window() -> Iterator[DriveWindow]:
    window = DriveWindow(CIRCLE, RacecarConfig(), "Circle")
    yield window
    pygame.display.quit()


def key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=0)


def drive(window: DriveWindow, keys: HeldKeys, seconds: float) -> Snapshot:
    for _ in range(round(seconds / DT)):
        window.advance(DT, keys)
    return window.world.snapshot


def test_the_car_starts_on_the_grid_at_rest(window: DriveWindow) -> None:
    snapshot = window.world.snapshot

    assert len(snapshot.cars) == 1
    assert snapshot.cars.speed[0] == 0.0
    assert snapshot.race.checkpoint[0] == -1
    assert window.renderer.mode is CameraMode.FOLLOW


def test_the_keys_drive_the_car(window: DriveWindow) -> None:
    snapshot = drive(window, HeldKeys(throttle=True), 2.0)

    assert snapshot.cars.speed[0] > 10
    assert snapshot.race.distance[0] > 10
    assert snapshot.time == pytest.approx(2.0)


def test_real_time_runs_one_decision_per_decision_time(window: DriveWindow) -> None:
    for _ in range(6):
        window.advance(1 / 60, HeldKeys())  # six frames: a tenth of a second

    assert window.world.snapshot.time == pytest.approx(0.1)


def test_a_long_stall_is_skipped_rather_than_played_in_a_burst(window: DriveWindow) -> None:
    window.advance(5.0, HeldKeys(throttle=True))  # the window was dragged for 5 s

    assert window.world.snapshot.time == pytest.approx(0.25)


def test_r_restarts_on_the_grid(window: DriveWindow) -> None:
    start = window.world.snapshot.cars.position.copy()
    drive(window, HeldKeys(throttle=True, left=True), 2.0)

    window.handle(key(pygame.K_r))

    restarted = window.world.snapshot
    np.testing.assert_array_equal(restarted.cars.position, start)
    assert restarted.race.distance[0] == 0.0
    window.agent.keys = HeldKeys()
    assert window.agent.act(np.zeros((1, 0), dtype=np.float32))[0, 0] == 0.0  # wheel straight


def test_p_pauses_the_race(window: DriveWindow) -> None:
    window.handle(key(pygame.K_p))
    drive(window, HeldKeys(throttle=True), 1.0)
    assert window.world.snapshot.time == 0.0

    window.handle(key(pygame.K_p))
    drive(window, HeldKeys(throttle=True), 1.0)
    assert window.world.snapshot.time == pytest.approx(1.0)


def test_c_switches_the_camera_and_number_keys_the_overlays(window: DriveWindow) -> None:
    window.handle(key(pygame.K_c))
    window.handle(key(pygame.K_2))

    assert window.renderer.mode is CameraMode.OVERVIEW
    assert window.renderer.overlays == {Overlay.CHECKPOINTS}


@pytest.mark.parametrize(
    "event",
    [pygame.event.Event(pygame.QUIT), key(pygame.K_ESCAPE)],
)
def test_closing_or_escape_stops_the_window(window: DriveWindow, event: pygame.event.Event) -> None:
    window.handle(event)

    assert not window.running


def test_the_mouse_wheel_zooms_and_dragging_pans(window: DriveWindow) -> None:
    snapshot = window.world.snapshot
    scale = window.renderer.camera(snapshot).scale

    window.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=2))
    assert window.renderer.camera(snapshot).scale == pytest.approx(scale * 1.15**2)

    window.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(10, 10)))
    window.handle(
        pygame.event.Event(pygame.MOUSEMOTION, rel=(30, 0), pos=(40, 10), buttons=(1, 0, 0))
    )
    window.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(40, 10)))
    window.handle(
        pygame.event.Event(pygame.MOUSEMOTION, rel=(30, 0), pos=(70, 10), buttons=(0, 0, 0))
    )
    assert window.renderer.mode is CameraMode.FREE


def test_a_resized_window_draws_at_its_new_size(window: DriveWindow) -> None:
    pygame.display.set_mode((400, 300), pygame.RESIZABLE)

    window.handle(pygame.event.Event(pygame.VIDEORESIZE, size=(400, 300), w=400, h=300))
    window.draw()

    assert window.renderer.size == (400, 300)


def test_the_caption_shows_the_best_lap_this_session(window: DriveWindow) -> None:
    window.draw()
    assert pygame.display.get_caption()[0] == "Circle - best this session -:--.--- - Drive"

    window.session_best = 83.25
    window.handle(key(pygame.K_p))
    window.draw()

    assert pygame.display.get_caption()[0] == "Circle - best this session 1:23.250 - paused - Drive"


def keys_for_a_lap(snapshot: Snapshot) -> HeldKeys:
    """A person holding arrow keys to drive round the middle of the circle at about 12 m/s."""
    race, speed = snapshot.race, snapshot.cars.speed[0]
    # Point back towards the middle of the road; steer whichever way the car is off from that.
    error = race.heading_error[0] + 0.1 * race.offset[0]
    return HeldKeys(
        left=bool(error < -0.02),
        right=bool(error > 0.02),
        throttle=bool(speed < 11.5),
        brake=bool(speed > 13.0),
    )


def test_a_lap_can_be_driven_with_the_keys_and_its_time_is_kept(window: DriveWindow) -> None:
    while math.isnan(window.session_best):
        assert window.world.snapshot.time < 90, "no valid lap in 90 s"
        window.advance(DT, keys_for_a_lap(window.world.snapshot))
        window.draw()

    lap = CIRCLE.length / 12.0  # about 31 s
    assert window.session_best == pytest.approx(lap, rel=0.1)
    assert window.world.snapshot.race.best_lap[0] == window.session_best

    window.handle(key(pygame.K_r))  # a restart keeps the session's best
    assert window.session_best == pytest.approx(lap, rel=0.1)


def test_run_returns_when_the_window_is_closed(window: DriveWindow) -> None:
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    window.run()

    assert not pygame.display.get_init()


def test_arrow_keys_and_wasd_both_drive() -> None:
    def pressed(*codes: int) -> HeldKeys:
        return held_keys(lambda code: code in codes)

    assert pressed(pygame.K_UP, pygame.K_LEFT) == HeldKeys(left=True, throttle=True)
    assert pressed(pygame.K_s, pygame.K_d) == HeldKeys(right=True, brake=True)
    assert pressed() == HeldKeys()


def test_other_keys_do_nothing(window: DriveWindow) -> None:
    window.handle(key(pygame.K_x))

    assert window.running
    assert not window.paused
    assert window.renderer.mode is CameraMode.FOLLOW
    assert window.renderer.overlays == set()


def test_run_drive_opens_a_window_and_returns_when_it_is_closed() -> None:
    pygame.display.init()
    pygame.event.post(pygame.event.Event(pygame.QUIT))

    run_drive(CIRCLE, RacecarConfig(), "Circle")

    assert not pygame.display.get_init()


def test_at_speed_a_held_key_turns_the_wheels_only_a_little(window: DriveWindow) -> None:
    drive(window, HeldKeys(throttle=True), 2.0)
    speed = window.world.snapshot.cars.speed[0]

    cars = drive(window, HeldKeys(throttle=True, left=True), 0.3).cars

    max_steer = window.world.model.params.max_steer
    assert speed > 15  # m/s
    assert 0 < cars.steer[0] < 0.3 * max_steer  # a slow car's keys would reach full lock
