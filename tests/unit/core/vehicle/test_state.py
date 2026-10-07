"""Tests for mlracecar.core.vehicle.state."""

import numpy as np
import pytest

from mlracecar.core.vehicle.state import VehicleState


def test_cars_at_rest_stand_still_with_straight_wheels() -> None:
    state = VehicleState.at_rest([[1.0, 2.0], [3.0, 4.0]], [0.5, 4.0])

    assert len(state) == 2
    np.testing.assert_array_equal(state.position, [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(state.yaw, [0.5, 4.0 - 2 * np.pi])  # headings in [-pi, pi)
    for still in (state.vx, state.vy, state.yaw_rate, state.steer, state.speed):
        np.testing.assert_array_equal(still, [0.0, 0.0])


def test_cars_at_rest_need_a_heading_each() -> None:
    with pytest.raises(ValueError, match="2 positions but 1 headings"):
        VehicleState.at_rest([[0.0, 0.0], [1.0, 0.0]], [0.0])


def test_speed_counts_sideways_motion_too() -> None:
    state = VehicleState.at_rest([[0.0, 0.0]], [0.0])
    moving = VehicleState(**{**vars(state), "vx": np.array([3.0]), "vy": np.array([4.0])})

    np.testing.assert_array_equal(moving.speed, [5.0])


@pytest.mark.parametrize(
    ("cars", "expected_x"),
    [
        ([2, 0], [2.0, 0.0]),
        (1, [1.0]),
        (slice(1, None), [1.0, 2.0]),
        (np.array([True, False, True]), [0.0, 2.0]),
    ],
)
def test_select_picks_cars(cars: object, expected_x: list[float]) -> None:
    state = VehicleState.at_rest([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [0.0, 0.1, 0.2])

    picked = state.select(cars)  # type: ignore[arg-type]

    np.testing.assert_array_equal(picked.x, expected_x)
    np.testing.assert_array_equal(picked.yaw, np.array(expected_x) / 10)
    assert len(picked.steer) == len(expected_x)
