"""Tests for mlracecar.env.rewards: every term of the reward, and adding them up per run."""

from dataclasses import replace

import numpy as np
import pytest

from mlracecar.config.models import REWARD_TERMS, RewardConfig, SimulationConfig, VehicleConfig
from mlracecar.core.race.events import LapCompleted, OffTrack, RaceEvent
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.env.rewards import RewardFunction, Rewards, RewardTally

CAR = VehicleConfig().to_params()
TIMING = SimulationConfig().to_timing()
DT = TIMING.decision_dt
ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 48)
STILL = np.zeros((2, 2))


def start() -> Snapshot:
    """Two cars on the grid of a round track."""
    return World(CIRCLE, KinematicBicycle(CAR), TIMING, 2, np.random.default_rng(0)).snapshot


def later(snapshot: Snapshot, events: tuple[RaceEvent, ...] = (), **race: np.ndarray) -> Snapshot:
    """The same snapshot with some of the race changed and these events."""
    return replace(snapshot, race=replace(snapshot.race, **race), events=events)


def weights(**chosen: float) -> RewardConfig:
    """Every weight 0 except the ones given."""
    return RewardConfig.model_validate(dict.fromkeys(REWARD_TERMS, 0.0) | chosen)


def score(
    config: RewardConfig,
    before: Snapshot,
    after: Snapshot,
    actions: np.ndarray = STILL,
    previous: np.ndarray = STILL,
) -> Rewards:
    return RewardFunction(config, DT)(before, after, actions, previous)


def lap(car: int, *, valid: bool) -> LapCompleted:
    return LapCompleted(car=car, time=46.0, sectors=(15.0, 15.0, 16.0), valid=valid, at=50.0)


def off_track(car: int) -> OffTrack:
    return OffTrack(car=car, arc_length=100.0, at=10.0)


# --------------------------------------------------------------------------- #
# Each term
# --------------------------------------------------------------------------- #


def test_progress_scores_the_metres_gained_along_the_lap() -> None:
    before = later(start(), distance=np.array([0.0, 10.0]))
    after = later(before, distance=np.array([30.0, 5.0]))  # the second car went backwards

    rewards = score(weights(progress=0.1), before, after)

    np.testing.assert_allclose(rewards.terms["progress"], [3.0, -0.5])


def test_leaving_the_road_costs_points_each_time() -> None:
    before = start()
    after = later(before, events=(off_track(1), off_track(1)))  # left, came back, left again

    rewards = score(weights(off_track=10.0), before, after)

    np.testing.assert_allclose(rewards.terms["off_track"], [0.0, -20.0])


def test_time_costs_points_every_step() -> None:
    rewards = score(weights(time=2.0), start(), start())

    np.testing.assert_allclose(rewards.terms["time"], [-2.0 * DT] * 2)


def test_driving_the_wrong_way_costs_points_while_it_lasts() -> None:
    before = start()
    after = later(before, wrong_way=np.array([True, False]))

    rewards = score(weights(wrong_way=2.0), before, after)

    np.testing.assert_allclose(rewards.terms["wrong_way"], [-2.0 * DT, 0.0])


def test_changing_the_controls_costs_the_square_of_the_change() -> None:
    actions = np.array([[1.0, 0.0], [0.2, 0.3]])
    previous = np.array([[-1.0, 0.5], [0.2, 0.3]])

    rewards = score(weights(smoothness=1.0), start(), start(), actions, previous)

    np.testing.assert_allclose(rewards.terms["smoothness"], [-(2.0**2 + 0.5**2), 0.0])


def test_controls_are_clipped_before_the_change_is_measured() -> None:
    actions = np.array([[3.0, 0.0], [0.0, 0.0]])  # steering beyond full lock counts as full lock
    previous = np.array([[-1.0, 0.0], [0.0, 0.0]])

    rewards = score(weights(smoothness=1.0), start(), start(), actions, previous)

    np.testing.assert_allclose(rewards.terms["smoothness"], [-4.0, 0.0])


def test_only_valid_laps_earn_the_bonus() -> None:
    before = start()
    after = later(before, events=(lap(0, valid=True), lap(1, valid=False)))

    rewards = score(weights(lap=5.0), before, after)

    np.testing.assert_allclose(rewards.terms["lap"], [5.0, 0.0])


@pytest.mark.parametrize("actions", [np.zeros((3, 2)), np.full((2, 2), np.nan)])
def test_impossible_actions_are_refused(actions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="actions"):
        score(RewardConfig(), start(), start(), actions)


# --------------------------------------------------------------------------- #
# Putting them together
# --------------------------------------------------------------------------- #


def test_the_reward_is_the_sum_of_every_term_reported_in_order() -> None:
    before = later(start(), distance=np.array([0.0, 0.0]))
    after = later(
        before,
        events=(off_track(0), lap(1, valid=True)),
        distance=np.array([20.0, 40.0]),
        wrong_way=np.array([False, True]),
    )
    config = RewardConfig(
        progress=0.1, off_track=10.0, time=1.0, wrong_way=1.0, smoothness=1.0, lap=3.0
    )

    rewards = score(config, before, after, np.array([[0.5, 0.0], [0.0, 0.0]]))

    assert tuple(rewards.terms) == REWARD_TERMS
    np.testing.assert_allclose(rewards.total, np.sum(list(rewards.terms.values()), axis=0))
    np.testing.assert_allclose(rewards.total, [2.0 - 10.0 - DT - 0.25, 4.0 - DT - DT + 3.0])


def test_by_default_only_progress_and_leaving_the_road_count() -> None:
    before = later(start(), distance=np.array([0.0, 0.0]))
    after = later(
        before,
        events=(off_track(0), lap(1, valid=True)),
        distance=np.array([20.0, 40.0]),
        wrong_way=np.array([True, True]),
    )

    rewards = score(RewardConfig(), before, after, np.ones((2, 2)))

    np.testing.assert_allclose(rewards.total, [2.0 - 10.0, 4.0])
    assert not rewards.terms["time"].any()
    assert not rewards.terms["lap"].any()


def test_over_a_run_progress_adds_up_to_the_distance_driven() -> None:
    world = World(CIRCLE, KinematicBicycle(CAR), TIMING, 2, np.random.default_rng(0))
    reward = RewardFunction(RewardConfig(), DT)
    tally = RewardTally(2)
    actions = np.array([[0.1, 1.0], [0.1, 0.5]])  # full and half throttle, a gentle left
    for _ in range(100):
        before = world.snapshot
        tally.add(reward(before, world.step(actions), actions, actions))

    np.testing.assert_allclose(tally.sums["progress"], 0.1 * world.snapshot.race.distance)
    assert tally.sums["progress"][0] > tally.sums["progress"][1] > 0


def test_a_car_driving_off_the_road_loses_the_points_on_the_step_it_leaves() -> None:
    world = World(CIRCLE, KinematicBicycle(CAR), TIMING, 1, np.random.default_rng(0))
    reward = RewardFunction(RewardConfig(), DT)
    straight = np.array([[0.0, 1.0]])  # straight on, while the road curves away to the left
    penalties = []
    for _ in range(200):
        before = world.snapshot
        penalties.append(
            reward(before, world.step(straight), straight, straight).terms["off_track"][0]
        )

    assert penalties.count(-10.0) == 1
    assert set(penalties) == {0.0, -10.0}


# --------------------------------------------------------------------------- #
# Adding up a run
# --------------------------------------------------------------------------- #


def test_the_tally_adds_each_term_per_car_and_clears_the_cars_asked() -> None:
    tally = RewardTally(2)
    rewards = Rewards(
        total=np.array([1.5, -9.0]),
        terms=dict.fromkeys(REWARD_TERMS, np.zeros(2))
        | {"progress": np.array([1.5, 1.0]), "off_track": np.array([0.0, -10.0])},
    )

    tally.add(rewards)
    tally.add(rewards)

    assert tally.of(1) == dict.fromkeys(REWARD_TERMS, 0.0) | {"progress": 2.0, "off_track": -20.0}
    np.testing.assert_allclose(tally.total, [3.0, -18.0])

    tally.clear(np.array([False, True]))

    np.testing.assert_allclose(tally.total, [3.0, 0.0])
    tally.clear()
    np.testing.assert_allclose(tally.total, [0.0, 0.0])
