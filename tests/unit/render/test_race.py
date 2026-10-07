"""Tests for mlracecar.render.race: drawing a race offscreen, cameras, kerbs, and overlays."""

import math
from dataclasses import replace

import numpy as np
import pygame
import pytest

from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.geometry import FloatArray
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.render.camera import Camera
from mlracecar.render.race import (
    CAMERA_LAG,
    CAR_COLORS,
    CENTERLINE,
    EASE_GAP,
    FOLLOW_MARGIN,
    FOLLOW_SCALE,
    HUD_SPACE,
    KERB_RED,
    KERB_WHITE,
    KERB_WIDTH,
    NEXT_CHECKPOINT,
    RAY,
    VELOCITY,
    VIEW_AHEAD,
    CameraMode,
    Overlay,
    RaceRenderer,
    _kerb_blocks,
    interpolated,
)

ANGLES = np.linspace(0, 2 * np.pi, 16, endpoint=False)
OVAL = Track.build(np.column_stack([120 * np.cos(ANGLES), 60 * np.sin(ANGLES)]), [12.0] * 16)
CAR = VehicleConfig().to_params()
SIZE = (640, 400)


def race(cars: int = 4, decisions: int = 40) -> Snapshot:
    """``cars`` cars a couple of seconds after the start, at full throttle."""
    world = World(
        OVAL, KinematicBicycle(CAR), SimulationConfig().to_timing(), cars, np.random.default_rng(0)
    )
    for _ in range(decisions):
        world.step(np.tile([0.0, 1.0], (cars, 1)))
    return world.snapshot


def renderer() -> RaceRenderer:
    return RaceRenderer(OVAL, (CAR.length, CAR.width), SIZE)


def has_color(image: np.ndarray, color: tuple[int, int, int]) -> bool:
    """Whether any pixel is this colour, or nearly (smoothed lines blend at their edges)."""
    distance = np.abs(image.astype(int) - color).sum(axis=2)
    return bool((distance <= 24).any())


def test_a_frame_is_an_rgb_image_of_the_renderer_size() -> None:
    image = renderer().render(race())

    assert image.shape == (400, 640, 3)
    assert image.dtype == np.uint8


def test_the_follow_camera_centres_a_standing_car() -> None:
    drawing, snapshot = renderer(), race(decisions=0)

    for car in (0, 2):
        drawing.followed = car
        image = drawing.render(snapshot)
        assert tuple(image[200, 320]) == CAR_COLORS[car]


def moving_at(speed: float) -> Snapshot:
    """One car on the grid, rolling straight ahead at ``speed`` m/s."""
    snapshot = race(cars=1, decisions=0)
    return replace(snapshot, cars=replace(snapshot.cars, vx=np.array([speed])))


def test_the_follow_camera_looks_ahead_of_a_moving_car() -> None:
    drawing, snapshot = renderer(), moving_at(20.0)

    camera = drawing.camera(snapshot)

    ahead = np.asarray(camera.center) - snapshot.cars.position[0]
    heading = np.array([np.cos(snapshot.cars.yaw[0]), np.sin(snapshot.cars.yaw[0])])
    np.testing.assert_allclose(ahead, heading * 20.0 * VIEW_AHEAD / 2)
    assert camera.scale == FOLLOW_SCALE  # 25 m ahead fits at the chosen zoom


def test_the_follow_camera_zooms_out_to_keep_the_road_ahead_on_screen() -> None:
    for speed in (40.0, 80.0):
        drawing, snapshot = renderer(), moving_at(speed)
        camera = drawing.camera(snapshot)
        far_ahead = snapshot.cars.position[0] + (camera.center - snapshot.cars.position[0]) * 2
        pixel = camera.to_screen(far_ahead)
        assert camera.scale < FOLLOW_SCALE
        assert pixel.min() >= FOLLOW_MARGIN - 1e-6
        assert (pixel <= np.asarray(SIZE) - FOLLOW_MARGIN + 1e-6).all()
        car = camera.to_screen(snapshot.cars.position[0])
        assert (car >= FOLLOW_MARGIN - 1e-6).all()  # and the car itself still shows
        assert (car <= np.asarray(SIZE) - FOLLOW_MARGIN + 1e-6).all()


def lead(camera: Camera, snapshot: Snapshot) -> FloatArray:
    """How far ahead of the followed car the camera is centred, in metres."""
    ahead: FloatArray = np.asarray(camera.center) - snapshot.cars.position[0]
    return ahead


def test_braking_eases_the_camera_back_instead_of_jumping() -> None:
    drawing = renderer()
    drawing.camera(moving_at(40.0))
    braked = replace(moving_at(10.0), time=0.1)  # 30 m/s slower a tenth of a second later

    camera = drawing.camera(braked)

    before, target = 40.0 * VIEW_AHEAD / 2, 10.0 * VIEW_AHEAD / 2
    eased = before + (target - before) * (1 - math.exp(-0.1 / CAMERA_LAG))
    assert np.hypot(*lead(camera, braked)) == pytest.approx(eased)
    assert eased > 0.8 * before  # it has barely started to move


def test_the_camera_follows_the_heading_not_the_flick_of_the_steering() -> None:
    drawing, snapshot = renderer(), moving_at(30.0)
    sliding = replace(snapshot, cars=replace(snapshot.cars, vy=np.array([5.0])))

    ahead = lead(drawing.camera(sliding), sliding)

    heading = sliding.cars.yaw[0]
    np.testing.assert_allclose(ahead / np.hypot(*ahead), [np.cos(heading), np.sin(heading)])


@pytest.mark.parametrize("change", ["pause", "back in time", "other car"])
def test_the_camera_jumps_into_place_when_easing_makes_no_sense(change: str) -> None:
    drawing, fast = renderer(), race(cars=2, decisions=0)
    moving = replace(fast, cars=replace(fast.cars, vx=np.array([40.0, 10.0])), time=1.0)
    drawing.camera(moving)
    later = replace(moving, cars=replace(moving.cars, vx=np.array([10.0, 10.0])))
    if change == "pause":
        later = replace(later, time=1.0 + 2 * EASE_GAP)
    elif change == "back in time":
        later = replace(later, time=0.5)  # restarted
    else:
        drawing.followed = 1
        later = replace(later, time=1.05)

    camera = drawing.camera(later)

    car = drawing.followed
    ahead = np.asarray(camera.center) - later.cars.position[car]
    assert np.hypot(*ahead) == pytest.approx(10.0 * VIEW_AHEAD / 2)


def test_the_overview_shows_the_whole_track_clear_of_the_hud() -> None:
    drawing = renderer()
    drawing.mode = CameraMode.OVERVIEW

    edges = drawing.camera(race()).to_screen(np.concatenate([OVAL.left, OVAL.right]))

    assert edges[:, 0].min() >= HUD_SPACE
    assert (edges <= SIZE).all()
    assert (edges >= 0).all()


def test_zoomed_far_out_cars_are_dots_so_they_still_show() -> None:
    drawing, snapshot = renderer(), race(cars=1)
    drawing.mode = CameraMode.OVERVIEW
    drawing.zoom_at((320, 200), 0.1)  # cars a fraction of a pixel long

    image = drawing.render(snapshot)

    position = drawing.camera(snapshot).to_screen(snapshot.cars.position[0]).round().astype(int)
    assert tuple(image[position[1], position[0]]) == CAR_COLORS[0]


def test_the_camera_cycles_follow_overview_free() -> None:
    drawing = renderer()

    assert [drawing.next_camera() for _ in range(3)] == [
        CameraMode.OVERVIEW,
        CameraMode.FREE,
        CameraMode.FOLLOW,
    ]


def test_zooming_the_follow_camera_keeps_following() -> None:
    drawing, snapshot = renderer(), race(decisions=0)

    drawing.zoom_at((0, 0), 2.0)

    camera = drawing.camera(snapshot)
    assert drawing.mode is CameraMode.FOLLOW
    assert camera.scale == pytest.approx(12.0)
    np.testing.assert_allclose(camera.center, snapshot.cars.position[0])


@pytest.mark.parametrize("move", ["zoom", "pan"])
def test_moving_the_overview_frees_the_camera_from_where_it_was(move: str) -> None:
    drawing, snapshot = renderer(), race()
    drawing.mode = CameraMode.OVERVIEW
    overview = drawing.camera(snapshot)

    if move == "zoom":
        drawing.zoom_at((320, 200), 2.0)
        expected = overview.zoom_at((320, 200), 2.0)
    else:
        drawing.pan(50, -20)
        expected = overview.pan(50, -20)

    assert drawing.mode is CameraMode.FREE
    assert drawing.camera(snapshot) == expected


def test_panning_from_the_follow_camera_frees_it_too() -> None:
    drawing = renderer()

    drawing.pan(10, 10)

    assert drawing.mode is CameraMode.FREE


def test_a_resized_renderer_draws_at_the_new_size() -> None:
    drawing = renderer()

    drawing.resize((320, 240))

    assert drawing.render(race()).shape == (240, 320, 3)
    drawing.mode = CameraMode.OVERVIEW
    assert drawing.camera(race()).size == (320, 240)


def test_kerbs_line_the_tight_bends_only() -> None:
    blocks, red = _kerb_blocks(OVAL)

    centres = blocks.mean(axis=1)
    # The oval's ends bend at a 30 m radius, its sides at 240 m: kerbs only at the ends.
    assert (np.abs(centres[:, 0]) > 60).all()
    assert red.any()
    assert (~red).any()


def test_kerbs_are_drawn_when_close_enough_to_see() -> None:
    drawing, snapshot = renderer(), race()
    drawing.mode = CameraMode.OVERVIEW
    end_of_oval = drawing.camera(snapshot).to_screen((120.0, 0.0))
    drawing.zoom_at((float(end_of_oval[0]), float(end_of_oval[1])), 5.0)

    image = drawing.render(snapshot)

    assert drawing.camera(snapshot).scale * KERB_WIDTH > 5  # kerbs several pixels wide
    assert has_color(image, KERB_RED)
    assert has_color(image, KERB_WHITE)


def overlay_drawn(
    overlay: Overlay, color: tuple[int, int, int], **rays: object
) -> tuple[bool, bool]:
    """Whether the overlay's colour shows without and with it switched on."""
    drawing, snapshot = renderer(), race()
    before = has_color(drawing.render(snapshot, **rays), color)  # type: ignore[arg-type]
    assert drawing.toggle(overlay)
    return before, has_color(drawing.render(snapshot, **rays), color)  # type: ignore[arg-type]


def test_the_centerline_overlay_draws_the_middle_of_the_road() -> None:
    assert overlay_drawn(Overlay.CENTERLINE, CENTERLINE) == (False, True)


def test_the_checkpoint_overlay_highlights_the_next_checkpoint() -> None:
    assert overlay_drawn(Overlay.CHECKPOINTS, NEXT_CHECKPOINT) == (False, True)


def test_the_velocity_overlay_draws_arrows() -> None:
    assert overlay_drawn(Overlay.VELOCITY, VELOCITY) == (False, True)


def test_the_ray_overlay_draws_rays_when_there_are_some() -> None:
    snapshot = race()
    ends = snapshot.cars.position[:, None, :] + [[[10.0, 10.0], [-10.0, 10.0]]]

    assert overlay_drawn(Overlay.RAYS, RAY, rays=ends) == (False, True)
    assert overlay_drawn(Overlay.RAYS, RAY) == (False, False)


def test_switching_an_overlay_twice_turns_it_off() -> None:
    drawing = renderer()

    assert drawing.toggle(Overlay.VELOCITY)
    assert not drawing.toggle(Overlay.VELOCITY)
    assert drawing.overlays == set()


def test_the_free_camera_zooms_where_it_is() -> None:
    drawing, snapshot = renderer(), race()
    drawing.pan(0, 0)
    before = drawing.camera(snapshot)

    drawing.zoom_at((100, 100), 3.0)

    assert drawing.camera(snapshot) == before.zoom_at((100, 100), 3.0)


def test_a_track_without_tight_bends_has_no_kerbs() -> None:
    gentle = Track.build(300 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 16)

    blocks, red = _kerb_blocks(gentle)

    assert blocks.shape[0] == red.shape[0] == 0
    drawing = RaceRenderer(gentle, (CAR.length, CAR.width), SIZE)
    world = World(
        gentle, KinematicBicycle(CAR), SimulationConfig().to_timing(), 1, np.random.default_rng(0)
    )
    assert not has_color(drawing.render(world.snapshot), KERB_RED)


def test_cars_are_drawn_between_two_snapshots() -> None:
    before, after = race(decisions=10), race(decisions=11)

    halfway = interpolated(before, after, 0.5)

    np.testing.assert_allclose(
        halfway.cars.position, (before.cars.position + after.cars.position) / 2
    )
    assert halfway.time == pytest.approx((before.time + after.time) / 2)
    assert halfway.race is after.race
    np.testing.assert_array_equal(
        interpolated(before, after, 0.0).cars.position, before.cars.position
    )
    np.testing.assert_array_equal(
        interpolated(before, after, 1.0).cars.position, after.cars.position
    )


def test_headings_blend_the_short_way_round() -> None:
    snapshot = race(cars=1)
    pointing = [
        replace(snapshot, cars=replace(snapshot.cars, yaw=np.array([yaw]))) for yaw in (3.0, -3.0)
    ]

    halfway = interpolated(pointing[0], pointing[1], 0.5)

    assert abs(halfway.cars.yaw[0]) == pytest.approx(np.pi)  # through pi, not through 0


def test_snapshots_with_different_cars_are_not_blended() -> None:
    one, four = race(cars=1), race(cars=4)

    assert interpolated(one, four, 0.5) is four


def test_a_hint_joins_the_bottom_line() -> None:
    drawing, snapshot = renderer(), race()
    images = []
    for hint in ("", "R: restart  ·  C: camera"):
        surface = pygame.Surface(SIZE)
        drawing.draw(surface, snapshot, hint=hint)
        images.append(pygame.surfarray.array3d(surface).swapaxes(0, 1))
    plain, hinted = images

    bottom = slice(SIZE[1] - 40, SIZE[1])
    assert (plain[bottom] != hinted[bottom]).any()
    assert (plain[: SIZE[1] - 40] == hinted[: SIZE[1] - 40]).all()
