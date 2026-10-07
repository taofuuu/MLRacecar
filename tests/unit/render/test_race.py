"""Tests for mlracecar.render.race: drawing a race offscreen, cameras, kerbs, and overlays."""

import numpy as np
import pytest

from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.render.race import (
    CAR_COLORS,
    CENTERLINE,
    HUD_SPACE,
    KERB_RED,
    KERB_WHITE,
    KERB_WIDTH,
    NEXT_CHECKPOINT,
    RAY,
    VELOCITY,
    CameraMode,
    Overlay,
    RaceRenderer,
    _kerb_blocks,
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


def test_the_follow_camera_keeps_the_followed_car_in_the_middle() -> None:
    drawing, snapshot = renderer(), race()

    for car in (0, 2):
        drawing.followed = car
        image = drawing.render(snapshot)
        assert tuple(image[200, 320]) == CAR_COLORS[car]


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
    drawing, snapshot = renderer(), race()

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
