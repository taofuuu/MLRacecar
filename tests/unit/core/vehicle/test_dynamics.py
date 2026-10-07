"""Tests for mlracecar.core.vehicle.dynamics: actions and actuators shared by every car model."""

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import ArrayLike

from mlracecar.config.models import VehicleConfig
from mlracecar.core.vehicle.dynamics import (
    GRAVITY,
    checked_actions,
    next_speed,
    steer_toward,
)

CAR = VehicleConfig().to_params()
DT = 1 / 120


def test_actions_are_clipped_to_the_pedal_and_wheel_range() -> None:
    np.testing.assert_array_equal(
        checked_actions([[2.0, -3.0], [0.5, 0.25]], 2), [[1, -1], [0.5, 0.25]]
    )


@pytest.mark.parametrize("actions", [[[0.0, 0.0]], [[0.0, 0.0, 0.0]] * 2, [0.0, 0.0]])
def test_actions_need_two_numbers_per_car(actions: ArrayLike) -> None:
    with pytest.raises(ValueError, match=r"expected \[steer, pedal\] actions of shape \(2, 2\)"):
        checked_actions(actions, 2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_actions_must_be_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        checked_actions([[0.0, bad]], 1)


# --------------------------------------------------------------------------- #
# Steering
# --------------------------------------------------------------------------- #


def test_wheels_turn_toward_the_command_at_the_steering_rate() -> None:
    steer = np.array([0.0, 0.0, CAR.max_steer])

    turned = steer_toward(steer, np.array([1.0, 0.0001, -1.0]), CAR, DT)

    reach = CAR.steer_rate * DT
    np.testing.assert_allclose(turned, [reach, 0.0001 * CAR.max_steer, CAR.max_steer - reach])


def test_lock_to_lock_takes_twice_the_lock_over_the_rate() -> None:
    steer = np.array([CAR.max_steer])
    steps = 0
    while steer[0] > -CAR.max_steer:
        steer = steer_toward(steer, np.array([-1.0]), CAR, DT)
        steps += 1

    # Whole steps: the last one may finish the turn a fraction of a step late.
    assert 0 <= steps * DT - 2 * CAR.max_steer / CAR.steer_rate <= DT * (1 + 1e-9)
    assert steer[0] == pytest.approx(-CAR.max_steer)


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #


def acceleration(speed: float, pedal: float) -> float:
    return float((next_speed(np.array([speed]), np.array([pedal]), CAR, DT)[0] - speed) / DT)


def test_full_throttle_from_rest_pushes_with_the_engines_full_force() -> None:
    rolling = CAR.rolling_resistance * CAR.mass * GRAVITY

    assert acceleration(0.0, 1.0) == pytest.approx((CAR.max_drive_force - rolling) / CAR.mass)


def test_at_speed_the_engines_power_limits_its_push() -> None:
    speed = 50.0  # above max_power / max_drive_force = 25 m/s
    resistance = acceleration(speed, 0.0)

    engine = (acceleration(speed, 1.0) - resistance) * CAR.mass

    assert engine == pytest.approx(CAR.max_power / speed)
    assert (acceleration(speed, 0.5) - resistance) * CAR.mass == pytest.approx(engine / 2)


def test_braking_pushes_with_the_brakes_full_force() -> None:
    speed = 20.0

    brakes = (acceleration(speed, 0.0) - acceleration(speed, -1.0)) * CAR.mass

    assert brakes == pytest.approx(CAR.max_brake_force)


def test_air_drag_grows_with_the_square_of_speed() -> None:
    rolling = -CAR.rolling_resistance * GRAVITY  # the same slowing at every speed

    ratio = (acceleration(40.0, 0.0) - rolling) / (acceleration(20.0, 0.0) - rolling)

    assert ratio == pytest.approx(4.0)


@pytest.mark.parametrize("pedal", [0.0, -1.0, 0.001])
def test_a_stopped_car_stays_stopped_unless_the_engine_beats_the_tyres(pedal: float) -> None:
    assert next_speed(np.array([0.0]), np.array([pedal]), CAR, DT)[0] == 0.0


def test_braking_stops_the_car_without_reversing() -> None:
    assert next_speed(np.array([0.01]), np.array([-1.0]), CAR, DT)[0] == 0.0


def test_no_resistance_means_coasting_forever() -> None:
    frictionless = replace(CAR, drag_coefficient=0.0, rolling_resistance=0.0)

    assert next_speed(np.array([30.0]), np.array([0.0]), frictionless, DT)[0] == 30.0
