"""Tests for mlracecar.render.camera: converting between metres and pixels, panning, zooming."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlracecar.render.camera import DEFAULT_SCALE, MAX_SCALE, MIN_SCALE, Camera

world_points = st.tuples(st.floats(-5000, 5000), st.floats(-5000, 5000))
pixels = st.tuples(st.floats(0, 1280), st.floats(0, 800))
cameras = st.builds(
    Camera,
    center=world_points,
    scale=st.floats(MIN_SCALE, MAX_SCALE),
    size=st.tuples(st.integers(100, 2000), st.integers(100, 2000)),
)


def test_the_centre_of_the_world_view_is_the_centre_of_the_window() -> None:
    camera = Camera(center=(100.0, 50.0), scale=2.0, size=(800, 600))
    np.testing.assert_allclose(camera.to_screen((100.0, 50.0)), (400.0, 300.0))


def test_up_in_the_world_is_up_on_screen() -> None:
    camera = Camera(scale=2.0, size=(800, 600))
    np.testing.assert_allclose(
        camera.to_screen([(10.0, 0.0), (0.0, 10.0)]), [(420, 300), (400, 280)]
    )


@given(cameras, world_points)
def test_to_world_undoes_to_screen(camera: Camera, point: tuple[float, float]) -> None:
    np.testing.assert_allclose(camera.to_world(camera.to_screen(point)), point, atol=1e-6)


@pytest.mark.parametrize("scale", [0.0, -1.0, 1e-6, 1e6])
def test_zoom_stays_within_limits(scale: float) -> None:
    assert MIN_SCALE <= Camera(scale=scale).scale <= MAX_SCALE


def test_the_window_shows_the_area_around_the_centre() -> None:
    low, high = Camera(center=(10.0, 20.0), scale=2.0, size=(800, 600)).visible_area()
    np.testing.assert_allclose(low, (-190, -130))
    np.testing.assert_allclose(high, (210, 170))


@given(cameras, pixels, st.floats(-500, 500), st.floats(-500, 500))
def test_panning_moves_the_world_with_the_mouse(
    camera: Camera, pixel: tuple[float, float], dx: float, dy: float
) -> None:
    grabbed = camera.to_world(pixel)
    panned = camera.pan(dx, dy)
    np.testing.assert_allclose(panned.to_screen(grabbed), np.add(pixel, (dx, dy)), atol=1e-6)


@given(cameras, pixels, st.floats(0.1, 10))
def test_zooming_keeps_the_point_under_the_cursor_still(
    camera: Camera, pixel: tuple[float, float], factor: float
) -> None:
    under_cursor = camera.to_world(pixel)
    zoomed = camera.zoom_at(pixel, factor)
    np.testing.assert_allclose(zoomed.to_world(pixel), under_cursor, atol=1e-6)


def test_zoom_at_changes_the_scale() -> None:
    assert Camera(scale=2.0).zoom_at((0, 0), 1.5).scale == 3.0


def test_fit_shows_every_point_inside_the_margin() -> None:
    points = np.array([(-300.0, 10.0), (500.0, 90.0), (100.0, -40.0)])
    camera = Camera(size=(800, 600)).fit(points, margin=50)
    screen = camera.to_screen(points)
    assert np.all(screen >= 50 - 1e-9)
    assert np.all(screen <= np.array([750, 550]) + 1e-9)
    assert screen[:, 0].min() == pytest.approx(50)  # the wide side fills the window
    assert screen[:, 0].max() == pytest.approx(750)


def test_fit_with_nothing_to_show_resets_the_view() -> None:
    camera = Camera(center=(40.0, 40.0), scale=9.0).fit([])
    assert camera.center == (0.0, 0.0)
    assert camera.scale == DEFAULT_SCALE


def test_fit_on_a_single_point_zooms_in_as_far_as_allowed() -> None:
    camera = Camera().fit([(3.0, 4.0)])
    assert camera.center == (3.0, 4.0)
    assert camera.scale == MAX_SCALE


def test_resizing_keeps_the_view_centred() -> None:
    camera = Camera(center=(5.0, 6.0), scale=3.0).resized((300, 200))
    assert camera.size == (300, 200)
    np.testing.assert_allclose(camera.to_screen((5.0, 6.0)), (150, 100))


@pytest.mark.parametrize(
    ("scale", "step"),
    [(0.05, 500.0), (1.0, 20.0), (2.0, 10.0), (3.2, 5.0), (16.0, 1.0), (50.0, 0.5)],
)
def test_grid_step_is_a_round_number(scale: float, step: float) -> None:
    assert Camera(scale=scale).grid_step() == step


@given(st.floats(MIN_SCALE, MAX_SCALE))
def test_grid_lines_are_at_least_16_pixels_but_not_too_far_apart(scale: float) -> None:
    pixels_apart = Camera(scale=scale).grid_step() * scale
    assert 16 - 1e-9 <= pixels_apart < 16 * 2.5 + 1e-9
