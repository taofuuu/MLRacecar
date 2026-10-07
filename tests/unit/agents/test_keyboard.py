"""Tests for mlracecar.agents.keyboard: held keys become smooth steering and a pedal."""

import math

import numpy as np
import pytest

from mlracecar.agents.base import Agent
from mlracecar.agents.keyboard import (
    CENTRE_TIME,
    GRIP_MARGIN,
    STEER_TIME,
    HeldKeys,
    KeyboardAgent,
)
from mlracecar.config.models import VehicleConfig
from mlracecar.core.vehicle.dynamics import GRAVITY

DT = 0.05
ONE_CAR = np.zeros((1, 0), dtype=np.float32)


def steering(agent: KeyboardAgent, keys: HeldKeys, decisions: int) -> list[float]:
    agent.keys = keys
    return [float(agent.act(ONE_CAR)[0, 0]) for _ in range(decisions)]


def test_it_is_an_agent() -> None:
    agent: Agent = KeyboardAgent(DT)  # checked by mypy

    assert agent.act(ONE_CAR).shape == (1, 2)


def test_with_no_keys_the_car_coasts_straight() -> None:
    action = KeyboardAgent(DT).act(ONE_CAR)

    np.testing.assert_array_equal(action, [[0.0, 0.0]])
    assert action.dtype == np.float32


def test_holding_a_key_turns_the_wheel_to_full_lock_in_steer_time() -> None:
    agent = KeyboardAgent(DT)
    decisions = round(STEER_TIME / DT)

    left = steering(agent, HeldKeys(left=True), decisions + 2)

    assert left[0] == pytest.approx(DT / STEER_TIME)
    assert left[decisions - 1] == pytest.approx(1.0)
    assert left[-1] == 1.0  # and stays there
    assert steering(KeyboardAgent(DT), HeldKeys(right=True), decisions)[-1] == pytest.approx(-1.0)


def test_letting_go_straightens_the_wheel_faster() -> None:
    agent = KeyboardAgent(DT)
    steering(agent, HeldKeys(left=True), 10)

    back = steering(agent, HeldKeys(), round(CENTRE_TIME / DT))

    assert back[-1] == pytest.approx(0.0, abs=1e-6)
    assert CENTRE_TIME < STEER_TIME


def test_steering_the_other_way_goes_through_straight_quickly() -> None:
    agent = KeyboardAgent(DT)
    steering(agent, HeldKeys(left=True), 10)

    turning = steering(agent, HeldKeys(right=True), 10)

    straight = next(index for index, steer in enumerate(turning) if steer <= 0)
    assert straight == round(CENTRE_TIME / DT) - 1
    assert turning[-1] == pytest.approx(-1.0)


def test_both_steering_keys_cancel_out() -> None:
    assert steering(KeyboardAgent(DT), HeldKeys(left=True, right=True), 3) == [0.0] * 3


@pytest.mark.parametrize(
    ("keys", "pedal"),
    [
        (HeldKeys(throttle=True), 1.0),
        (HeldKeys(brake=True), -1.0),
        (HeldKeys(throttle=True, brake=True), -1.0),  # braking wins
        (HeldKeys(), 0.0),
    ],
)
def test_the_pedal_follows_the_keys_at_once(keys: HeldKeys, pedal: float) -> None:
    agent = KeyboardAgent(DT)
    agent.keys = keys

    assert agent.act(ONE_CAR)[0, 1] == pedal


def test_a_reset_straightens_the_wheel() -> None:
    agent = KeyboardAgent(DT)
    steering(agent, HeldKeys(left=True), 3)

    agent.reset(seed=7)
    agent.keys = HeldKeys()

    assert agent.act(ONE_CAR)[0, 0] == 0.0


def test_every_car_in_the_batch_gets_the_same_action() -> None:
    agent = KeyboardAgent(DT)
    agent.keys = HeldKeys(left=True, throttle=True)

    actions = agent.act(np.zeros((3, 5), dtype=np.float32))

    assert actions.shape == (3, 2)
    assert (actions == actions[0]).all()


# --------------------------------------------------------------------------- #
# Speed-sensitive steering
# --------------------------------------------------------------------------- #

CAR = VehicleConfig().to_params()


def grip_angle(speed: float) -> float:
    """The widest wheel angle the tyres can follow at this speed, in radians."""
    return math.atan(CAR.grip * GRAVITY * CAR.wheelbase / speed**2)


def test_slowly_the_keys_reach_full_lock() -> None:
    np.testing.assert_array_equal(KeyboardAgent(DT, CAR).reach(np.array([0.0, 5.0])), [1.0, 1.0])


def test_at_speed_the_keys_turn_only_about_as_far_as_the_tyres_can_use() -> None:
    agent = KeyboardAgent(DT, CAR)
    speeds = np.array([10.0, 20.0, 30.0, 50.0])

    reach = agent.reach(speeds)

    angles = [GRIP_MARGIN * grip_angle(speed) for speed in speeds]
    np.testing.assert_allclose(reach * CAR.max_steer, np.minimum(angles, CAR.max_steer))
    assert (np.diff(reach) < 0).all()  # the faster, the less


def test_without_the_car_the_keys_reach_full_lock_at_any_speed() -> None:
    assert KeyboardAgent(DT).reach(np.array([50.0]))[0] == 1.0


def test_a_held_key_at_speed_turns_as_far_as_the_keys_reach() -> None:
    agent = KeyboardAgent(DT, CAR)
    agent.keys = HeldKeys(left=True)
    fast = np.array([[30.0]], dtype=np.float32)

    actions = [agent.act(fast)[0, 0] for _ in range(round(STEER_TIME / DT))]

    reach = agent.reach(np.array([30.0]))[0]
    assert actions[0] == pytest.approx(reach * DT / STEER_TIME, rel=1e-6)  # a tap: a little
    assert actions[-1] == pytest.approx(reach, rel=1e-6)  # held: as far as the keys reach


def test_each_car_steers_for_its_own_speed() -> None:
    agent = KeyboardAgent(DT, CAR)
    agent.keys = HeldKeys(right=True, throttle=True)
    speeds = np.array([[0.0], [40.0]], dtype=np.float32)
    for _ in range(round(STEER_TIME / DT)):
        actions = agent.act(speeds)

    assert actions[0, 0] == pytest.approx(-1.0)
    assert actions[1, 0] == pytest.approx(-agent.reach(np.array([40.0]))[0], rel=1e-6)
    np.testing.assert_array_equal(actions[:, 1], [1.0, 1.0])
