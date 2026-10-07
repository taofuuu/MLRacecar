"""Tests for mlracecar.core.vehicle.kinematic: the kinematic bicycle model."""

import math
from dataclasses import fields, replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import ArrayLike

from mlracecar.config.models import VehicleConfig
from mlracecar.core.geometry import FloatArray, unit_vector
from mlracecar.core.vehicle.dynamics import AIR_DENSITY, GRAVITY, DynamicsModel
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.core.vehicle.state import VehicleState
from strategies import cars_and_actions

CAR = VehicleConfig().to_params()
MODEL = KinematicBicycle(CAR)
DT = 1 / 120


def one_car(*, speed: float = 0.0, steer: float = 0.0, yaw: float = 0.0) -> VehicleState:
    """A car at the origin, rolling straight ahead with its wheels at ``steer``."""
    state = VehicleState.at_rest([[0.0, 0.0]], [yaw])
    return replace(state, vx=np.array([speed]), steer=np.array([steer]))


def drive(
    state: VehicleState, actions: ArrayLike, steps: int, *, model: KinematicBicycle = MODEL
) -> list[VehicleState]:
    """The states after each of ``steps`` steps with the same actions throughout."""
    states = []
    for _ in range(steps):
        state = model.step(state, actions, DT)
        states.append(state)
    return states


def rear_axle(state: VehicleState) -> FloatArray:
    return state.position - CAR.wheelbase / 2 * unit_vector(state.yaw)


def fitted_circle(points: FloatArray) -> tuple[FloatArray, float]:
    """Least-squares circle through points of shape ``(K, 2)``: its centre and radius."""
    x, y = points.T
    terms = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(terms, x * x + y * y, rcond=None)
    return np.array([a, b]), float(np.sqrt(c + a * a + b * b))


def top_speed(params: VehicleParams) -> float:
    """Where the engine's push at full throttle equals the air and tyre resistance."""
    drag = 0.5 * AIR_DENSITY * params.drag_coefficient * params.frontal_area
    rolling = params.rolling_resistance * params.mass * GRAVITY
    # Above the power limit: max_power / v = drag v² + rolling, a cubic in v.
    roots = np.roots([drag, 0.0, rolling, -params.max_power])
    powered = max(root.real for root in roots if abs(root.imag) < 1e-9)
    if powered >= params.max_power / params.max_drive_force:
        return float(powered)
    # Otherwise the full force holds all the way: max_drive_force = drag v² + rolling.
    return math.sqrt((params.max_drive_force - rolling) / drag)


def test_it_is_a_dynamics_model() -> None:
    model: DynamicsModel = MODEL  # checked by mypy

    assert model.step(one_car(), [[0.0, 0.0]], DT).x[0] == 0.0


# --------------------------------------------------------------------------- #
# Acceptance criteria (#18)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("degrees", [5.0, 15.0, 30.0])
def test_constant_steering_drives_a_circle_of_radius_wheelbase_over_tan_steer(
    degrees: float,
) -> None:
    angle = math.radians(degrees)
    radius = CAR.wheelbase / math.tan(angle)
    # Half the speed at which the grip limit would widen the circle, held steady: with the
    # speed changing, the steps' small lag in direction would shift the circle as it goes.
    speed = 0.5 * math.sqrt(CAR.grip * GRAVITY * radius)
    steady = KinematicBicycle(replace(CAR, drag_coefficient=0.0, rolling_resistance=0.0))
    lap = 2 * math.pi * radius / speed
    start = one_car(speed=speed, steer=angle)
    states = drive(start, [[angle / CAR.max_steer, 0.0]], round(lap / DT), model=steady)

    points = np.concatenate([rear_axle(state) for state in states])
    centre, fitted = fitted_circle(points)

    assert fitted == pytest.approx(radius, rel=1e-3)  # the ticket asks for 1%
    np.testing.assert_allclose(np.hypot(*(points - centre).T), fitted, rtol=1e-9)
    # Turning left: the centre is level with the rear axle, on the left.
    np.testing.assert_allclose(centre, rear_axle(start)[0] + [0.0, radius], atol=0.01 * radius)


def test_full_throttle_reaches_the_top_speed_where_power_meets_drag() -> None:
    state = drive(one_car(), [[0.0, 1.0]], round(240 / DT))[-1]  # four minutes

    assert top_speed(CAR) * 3.6 == pytest.approx(303, abs=5)  # km/h
    assert state.speed[0] == pytest.approx(top_speed(CAR), rel=1e-4)


def test_at_top_speed_the_car_neither_gains_nor_loses_speed() -> None:
    state = drive(one_car(speed=top_speed(CAR)), [[0.0, 1.0]], 120)[-1]

    assert state.speed[0] == pytest.approx(top_speed(CAR), rel=1e-12)


def test_with_power_to_spare_the_top_speed_is_where_full_force_meets_drag() -> None:
    model = KinematicBicycle(replace(CAR, max_power=1e9))
    state = drive(one_car(speed=top_speed(model.params)), [[0.0, 1.0]], 120, model=model)[-1]

    assert top_speed(model.params) < model.params.max_power / model.params.max_drive_force
    assert state.speed[0] == pytest.approx(top_speed(model.params), rel=1e-12)


def test_a_car_alone_moves_exactly_as_it_does_among_1024() -> None:
    rng = np.random.default_rng(18)
    count, steps = 1024, 30
    batch = VehicleState(
        x=rng.uniform(-500, 500, count),
        y=rng.uniform(-500, 500, count),
        yaw=rng.uniform(-np.pi, np.pi, count),
        vx=rng.uniform(0, 80, count),
        vy=rng.uniform(-2, 2, count),
        yaw_rate=rng.uniform(-1, 1, count),
        steer=rng.uniform(-CAR.max_steer, CAR.max_steer, count),
    )
    actions = rng.uniform(-1.2, 1.2, (steps, count, 2))

    alone = [batch.select([car]) for car in range(count)]
    for step in range(steps):
        batch = MODEL.step(batch, actions[step], DT)
        alone = [MODEL.step(car, actions[step, [index]], DT) for index, car in enumerate(alone)]

    for field in fields(VehicleState):
        together = getattr(batch, field.name)
        apart = np.concatenate([getattr(car, field.name) for car in alone])
        np.testing.assert_array_equal(apart, together, err_msg=field.name)


# --------------------------------------------------------------------------- #
# How the car drives
# --------------------------------------------------------------------------- #


def test_with_straight_wheels_the_car_drives_straight_along_its_heading() -> None:
    heading = 0.7
    state = drive(one_car(speed=20.0, yaw=heading), [[0.0, 0.3]], 240)[-1]

    assert state.yaw[0] == heading
    assert state.vy[0] == state.yaw_rate[0] == 0.0
    direction = state.position[0] / np.linalg.norm(state.position[0])
    np.testing.assert_allclose(direction, unit_vector(heading), atol=1e-12)


def test_too_fast_for_the_turn_the_car_runs_wide_at_the_grip_limit() -> None:
    speed = 30.0
    state = MODEL.step(one_car(speed=speed, steer=CAR.max_steer), [[1.0, 0.0]], DT)

    # tan(sideslip) = tan(turn) / 2, so the rear axle's radius is wheelbase · vx / (2 vy).
    rear_radius = CAR.wheelbase * state.vx[0] / (2 * state.vy[0])
    assert rear_radius == pytest.approx(state.speed[0] ** 2 / (CAR.grip * GRAVITY))
    assert rear_radius > 10 * CAR.wheelbase / math.tan(CAR.max_steer)  # much wider than asked


def test_more_grip_turns_tighter() -> None:
    def radius(grip: float) -> float:
        model = KinematicBicycle(replace(CAR, grip=grip))
        state = model.step(one_car(speed=30.0, steer=CAR.max_steer), [[1.0, 0.0]], DT)
        return float(state.speed[0] / state.yaw_rate[0])

    assert radius(2.0) < radius(1.0) / 1.9


def test_braking_stops_the_car_and_holds_it() -> None:
    states = drive(one_car(speed=100 / 3.6), [[0.0, -1.0]], 360)

    stopped = next(index for index, state in enumerate(states) if state.speed[0] == 0.0)
    assert all(state.speed[0] == 0.0 for state in states[stopped:])
    assert states[-1].x[0] == states[stopped].x[0]
    # Under full braking, the brakes alone would stop it in v² / (2 · brake force / mass).
    assert 0 < states[-1].x[0] < (100 / 3.6) ** 2 / (2 * CAR.max_brake_force / CAR.mass)


def test_the_default_car_is_quick() -> None:
    states = drive(one_car(), [[0.0, 1.0]], round(10 / DT))
    seconds = DT * next(index for index, state in enumerate(states) if state.speed[0] >= 100 / 3.6)

    assert 2.5 < seconds < 4  # 0-100 km/h, like a hot sporty car


def test_bad_actions_are_refused() -> None:
    with pytest.raises(ValueError, match="finite"):
        MODEL.step(one_car(), [[math.nan, 0.0]], DT)


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #


@given(cars_and_actions(max_steer=CAR.max_steer), st.floats(1e-4, 0.1))
def test_every_step_obeys_the_cars_limits(cars: tuple[VehicleState, FloatArray], dt: float) -> None:
    state, actions = cars

    after = MODEL.step(state, actions, dt)

    for field in fields(VehicleState):
        assert np.isfinite(getattr(after, field.name)).all(), field.name
    assert (after.speed >= 0).all()
    assert (np.abs(after.steer) <= CAR.max_steer * (1 + 1e-12)).all()
    assert (np.abs(after.steer - state.steer) <= CAR.steer_rate * dt * (1 + 1e-9) + 1e-15).all()
    assert (after.speed * np.abs(after.yaw_rate) <= CAR.grip * GRAVITY * (1 + 1e-9)).all()
    assert ((after.yaw >= -np.pi) & (after.yaw < np.pi)).all()
    moved = np.hypot(after.x - state.x, after.y - state.y)
    np.testing.assert_allclose(moved, after.speed * dt, rtol=1e-9, atol=1e-9)


@given(cars_and_actions(max_steer=CAR.max_steer))
def test_steering_the_other_way_mirrors_the_motion(cars: tuple[VehicleState, FloatArray]) -> None:
    state, actions = cars
    mirror = VehicleState(
        x=state.x,
        y=-state.y,
        yaw=-state.yaw,
        vx=state.vx,
        vy=-state.vy,
        yaw_rate=-state.yaw_rate,
        steer=-state.steer,
    )

    after = MODEL.step(state, actions, DT)
    mirrored = MODEL.step(mirror, actions * [-1.0, 1.0], DT)

    np.testing.assert_allclose(mirrored.x, after.x, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(mirrored.y, -after.y, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(np.cos(mirrored.yaw), np.cos(after.yaw), atol=1e-12)
    np.testing.assert_allclose(np.sin(mirrored.yaw), -np.sin(after.yaw), atol=1e-12)
    np.testing.assert_allclose(mirrored.steer, -after.steer, atol=1e-15)
