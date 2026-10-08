"""Tests for mlracecar.core.world: timing, steps, resets, frozen snapshots, and determinism."""

from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from mlracecar.config.models import VehicleConfig
from mlracecar.core.geometry import BoolArray, FloatArray, wrap_angle
from mlracecar.core.race.rules import OffTrackPolicy, RaceSettings
from mlracecar.core.race.state import RaceState
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import GridLayout, Pose, Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.core.world import StartPosition, Timing, World, random_poses
from roads import road_coordinates

ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)
TRACK = Track.build(60 * np.column_stack([2 * np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 12)
CAR = VehicleConfig().to_params()
MODEL = KinematicBicycle(CAR)
TIMING = Timing(physics_hz=120, action_repeat=6)
GRID = TRACK.start_grid(8, GridLayout(car_length=CAR.length, car_width=CAR.width))


def make_world(cars: int = 8, seed: int = 0, start: StartPosition = StartPosition.GRID) -> World:
    return World(TRACK, MODEL, TIMING, cars, np.random.default_rng(seed), start)


def random_actions(rng: np.random.Generator, cars: int) -> FloatArray:
    return rng.uniform(-1.0, 1.0, (cars, 2))


def assert_same(first: Any, second: Any) -> None:
    """Two car or race states, bitwise equal, field by field."""
    for field in fields(first):
        np.testing.assert_array_equal(
            getattr(first, field.name), getattr(second, field.name), err_msg=field.name
        )


def subset[State: (VehicleState, RaceState)](state: State, cars: BoolArray) -> State:
    return type(state)(**{field.name: getattr(state, field.name)[cars] for field in fields(state)})


def assert_at_rest(cars: VehicleState) -> None:
    for still in (cars.vx, cars.vy, cars.yaw_rate, cars.steer):
        assert (still == 0.0).all()


def test_timing_turns_rates_into_step_lengths() -> None:
    timing = Timing(physics_hz=120, action_repeat=6)

    assert timing.dt == pytest.approx(1 / 120)
    assert timing.decision_dt == pytest.approx(0.05)  # 20 decisions a second


# --------------------------------------------------------------------------- #
# Starting and stepping
# --------------------------------------------------------------------------- #


def test_cars_start_at_rest_on_their_own_grid_spots() -> None:
    world = make_world()

    snapshot = world.snapshot
    assert (snapshot.tick, snapshot.time, len(world)) == (0, 0.0, 8)
    np.testing.assert_array_equal(snapshot.cars.position, GRID.position)
    np.testing.assert_array_equal(snapshot.cars.yaw, wrap_angle(GRID.heading))
    assert_at_rest(snapshot.cars)


def test_a_world_needs_a_car() -> None:
    with pytest.raises(ValueError, match="at least 1 car, got 0"):
        make_world(cars=0)


def test_a_step_holds_each_action_for_one_driver_decision() -> None:
    world = make_world()
    actions = random_actions(np.random.default_rng(1), 8)
    expected = world.snapshot.cars
    for _ in range(TIMING.action_repeat):
        expected = MODEL.step(expected, actions, TIMING.dt)

    snapshot = world.step(actions)

    assert snapshot is world.snapshot
    assert (snapshot.tick, snapshot.time) == (6, 0.05)
    assert_same(snapshot.cars, expected)


def test_a_step_refuses_bad_actions() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_world(cars=1).step([[np.nan, 0.0]])


def test_snapshots_are_frozen() -> None:
    world = make_world()
    first = world.step(np.tile([0.0, 1.0], (8, 1)))
    kept = VehicleState(
        **{field.name: getattr(first.cars, field.name).copy() for field in fields(VehicleState)}
    )

    world.step(np.tile([1.0, 1.0], (8, 1)))
    world.reset()

    assert_same(first.cars, kept)
    with pytest.raises(ValueError, match="read-only"):
        first.cars.x[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        first.race.distance[0] = 0.0


# --------------------------------------------------------------------------- #
# Acceptance criteria (#20)
# --------------------------------------------------------------------------- #


def race(seed: int) -> list[Snapshot]:
    """60 decisions with random actions, and resets to random spots and the grid on the way."""
    world = make_world(cars=16, seed=seed)
    choices = np.random.default_rng(1000 + seed)  # the "driver": the same for both runs
    snapshots = []
    for decision in range(60):
        snapshots.append(world.step(random_actions(choices, 16)))
        if decision % 10 == 9:
            start = StartPosition.RANDOM if decision % 20 == 9 else StartPosition.GRID
            snapshots.append(world.reset(choices.random(16) < 0.3, start=start))
    return snapshots


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_same_seed_and_actions_give_exactly_the_same_race(seed: int) -> None:
    first, second = race(seed), race(seed)

    assert len(first) == len(second) == 66
    for one, other in zip(first, second, strict=True):
        assert (one.tick, one.time) == (other.tick, other.time)
        assert_same(one.cars, other.cars)
        assert_same(one.race, other.race)
        assert repr(one.events) == repr(other.events)  # repr: NaN sector times compare equal


def test_another_seed_starts_cars_elsewhere() -> None:
    first, second = (make_world(seed=seed, start=StartPosition.RANDOM) for seed in (0, 1))

    assert not np.array_equal(first.snapshot.cars.x, second.snapshot.cars.x)


@pytest.mark.parametrize("start", list(StartPosition))
def test_resetting_some_cars_does_not_affect_the_others(start: StartPosition) -> None:
    reset, untouched = make_world(seed=5), make_world(seed=5)
    choices = np.random.default_rng(6)
    for _ in range(20):
        actions = random_actions(choices, 8)
        reset.step(actions)
        untouched.step(actions)
    chosen = np.array([True, False, False, True, False, True, False, False])

    after = reset.reset(chosen, start=start)

    assert_at_rest(after.cars.select(chosen))
    assert_same(after.cars.select(~chosen), untouched.snapshot.cars.select(~chosen))
    assert_same(subset(after.race, ~chosen), subset(untouched.snapshot.race, ~chosen))
    np.testing.assert_array_equal(after.race.checkpoint[chosen], -1)  # their race restarts
    np.testing.assert_array_equal(after.race.distance[chosen], 0.0)
    for _ in range(20):
        actions = random_actions(choices, 8)
        one, other = reset.step(actions), untouched.step(actions)
        assert_same(one.cars.select(~chosen), other.cars.select(~chosen))
        assert_same(subset(one.race, ~chosen), subset(other.race, ~chosen))


# --------------------------------------------------------------------------- #
# Resets
# --------------------------------------------------------------------------- #


def test_a_grid_reset_puts_cars_back_on_their_own_spots() -> None:
    world = make_world()
    for _ in range(40):
        world.step(np.tile([0.3, 1.0], (8, 1)))

    snapshot = world.reset([False, True, False, True, False, False, False, False])

    np.testing.assert_array_equal(snapshot.cars.position[[1, 3]], GRID.position[[1, 3]])
    assert (snapshot.cars.speed[[0, 2, 4, 5, 6, 7]] > 0).all()


def test_a_reset_without_a_mask_resets_every_car() -> None:
    world = make_world()
    world.step(np.tile([0.0, 1.0], (8, 1)))

    snapshot = world.reset()

    np.testing.assert_array_equal(snapshot.cars.position, GRID.position)
    assert_at_rest(snapshot.cars)


def test_the_clock_keeps_running_through_resets() -> None:
    world = make_world()
    world.step(np.zeros((8, 2)))

    assert world.reset(start=StartPosition.RANDOM).tick == TIMING.action_repeat


def test_a_reset_needs_one_entry_per_car() -> None:
    with pytest.raises(ValueError, match=r"mask of shape \(8,\), got \(3,\)"):
        make_world().reset([True, False, True])


def test_the_grid_has_a_spot_for_each_car() -> None:
    grid = make_world().grid

    np.testing.assert_array_equal(grid.position, GRID.position)
    np.testing.assert_array_equal(grid.heading, GRID.heading)


def test_cars_can_be_reset_to_places_of_our_choosing() -> None:
    world = make_world(cars=3)
    for _ in range(10):
        world.step(np.tile([0.0, 1.0], (3, 1)))
    moved = world.snapshot.cars
    pole = Pose(np.repeat(GRID.position[:1], 2, axis=0), np.repeat(GRID.heading[:1], 2))

    snapshot = world.reset([True, False, True], pose=pole)

    np.testing.assert_array_equal(snapshot.cars.position[[0, 2]], pole.position)
    np.testing.assert_array_equal(snapshot.cars.yaw[[0, 2]], pole.heading)
    assert_at_rest(snapshot.cars.select(np.array([0, 2])))
    np.testing.assert_array_equal(snapshot.cars.position[1], moved.position[1])
    assert snapshot.race.distance[[0, 2]].tolist() == [0.0, 0.0]


def test_a_chosen_place_is_needed_for_each_car_reset() -> None:
    pole = Pose(GRID.position[:1], GRID.heading[:1])

    with pytest.raises(ValueError, match="a pose for each of the 2 cars reset"):
        make_world(cars=3).reset([True, True, False], pose=pole)


def test_random_places_depend_only_on_the_generator() -> None:
    first = random_poses(TRACK, CAR.width, np.random.default_rng(5), 3)
    again = random_poses(TRACK, CAR.width, np.random.default_rng(5), 3)
    world = make_world(cars=3, seed=5, start=StartPosition.RANDOM)

    np.testing.assert_array_equal(first.position, again.position)
    np.testing.assert_array_equal(world.snapshot.cars.position, first.position)


def test_random_starts_put_the_whole_car_on_the_road_facing_the_way_round() -> None:
    cars = make_world(cars=500, start=StartPosition.RANDOM).snapshot.cars

    arc_length, offset = road_coordinates(TRACK, cars.position)

    room = (TRACK.width_at(arc_length) - CAR.width) / 2
    assert (np.abs(offset) <= room + 1e-9).all()
    heading_error = wrap_angle(cars.yaw - TRACK.pose_at(arc_length).heading)
    np.testing.assert_allclose(heading_error, 0.0, atol=1e-9)
    assert_at_rest(cars)
    # Spread over the whole lap and both sides of the road.
    sectors = np.histogram(arc_length % TRACK.length, bins=10, range=(0, TRACK.length))[0]
    assert (sectors > 20).all()
    assert (offset > room / 2).any()
    assert (offset < -room / 2).any()


# --------------------------------------------------------------------------- #
# Off the road
# --------------------------------------------------------------------------- #


def test_the_grass_slows_cars_down_unless_the_settings_say_otherwise() -> None:
    assert make_world().rules.settings == RaceSettings(off_track=OffTrackPolicy.SLOWDOWN)


def test_a_car_whose_run_is_over_stays_where_it_is_until_it_is_reset() -> None:
    settings = RaceSettings(off_track=OffTrackPolicy.TERMINATE)
    world = World(TRACK, MODEL, TIMING, 1, np.random.default_rng(0), settings=settings)
    while not world.snapshot.race.out[0]:  # full throttle, full left: off the road soon
        assert world.snapshot.time < 30, "never left the road"
        world.step([[1.0, 1.0]])
    stopped = world.snapshot
    assert stopped.cars.speed[0] == 0.0

    for _ in range(10):
        world.step([[0.0, 1.0]])
    assert_same(world.snapshot.cars, stopped.cars)

    world.reset()
    assert not world.snapshot.race.out[0]
    assert world.step([[0.0, 1.0]]).cars.speed[0] > 0


def test_a_snapshot_can_be_narrowed_to_some_cars() -> None:
    world = make_world(cars=3)
    world.step(np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 0.0]]))
    snapshot = world.snapshot

    some = snapshot.select(np.array([True, False, True]))

    assert (some.tick, some.time, some.events) == (snapshot.tick, snapshot.time, ())
    assert_same(some.cars, snapshot.cars.select([0, 2]))
    for field in fields(RaceState):
        np.testing.assert_array_equal(
            getattr(some.race, field.name), getattr(snapshot.race, field.name)[[0, 2]]
        )
