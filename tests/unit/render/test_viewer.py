"""Tests for mlracecar.render.viewer: drawing an environment's race, run off-screen."""

import numpy as np
import pygame
import pytest

from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.render.viewer import SIZE, RaceViewer

CAR = VehicleConfig().to_params()
ANGLES = np.linspace(0, 2 * np.pi, 24, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 24)
SNAPSHOT = World(
    CIRCLE, KinematicBicycle(CAR), SimulationConfig().to_timing(), 1, np.random.default_rng(0)
).snapshot


def test_pictures_are_rgb_images_of_the_viewer_size() -> None:
    frame = RaceViewer(CIRCLE, CAR, "rgb_array", 20).render(SNAPSHOT, None)

    assert frame is not None
    assert frame.shape == (SIZE[1], SIZE[0], 3)


def test_the_window_opens_on_first_use_and_closes() -> None:
    viewer = RaceViewer(CIRCLE, CAR, "human", 1000)

    assert viewer.render(SNAPSHOT, None) is None
    assert pygame.display.get_surface() is not None
    viewer.render(SNAPSHOT, None)  # the same window again

    viewer.close()
    viewer.close()  # closing twice is fine
    assert pygame.display.get_surface() is None


def test_unknown_modes_are_refused() -> None:
    with pytest.raises(ValueError, match="'human' or 'rgb_array'"):
        RaceViewer(CIRCLE, CAR, "ascii", 20)
