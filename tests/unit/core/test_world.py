"""Tests for mlracecar.core.world: timing, steps, resets, frozen snapshots, and determinism."""

from dataclasses import fields

import numpy as np
import pytest

from mlracecar.config.models import VehicleConfig
from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.track.model import GridLayout, Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.core.world import Snapshot, StartPosition, Timing, World
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


def assert_same_cars(first: VehicleState, second: VehicleState) -> None:
    """Bitwise equal, field by field."""
    for field in fields(VehicleState):
        np.testing.assert_array_equal(
            getattr(first, field.name), getattr(second, field.name), err_msg=field.name
        )


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
    assert_same_cars(snapshot.cars, expected)


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

    assert_same_cars(first.cars, kept)
    with pytest.raises(ValueError, match="read-only"):
        first.cars.x[0] = 0.0


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
        assert_same_cars(one.cars, other.cars)


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
    assert_same_cars(after.cars.select(~chosen), untouched.snapshot.cars.select(~chosen))
    for _ in range(20):
        actions = random_actions(choices, 8)
        assert_same_cars(
            reset.step(actions).cars.select(~chosen), untouched.step(actions).cars.select(~chosen)
        )


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
