"""Tests for mlracecar.core.race.rules: progress, checkpoints, laps, and timing."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import ArrayLike

from drivers import CenterlineDriver
from mlracecar.config.models import SimulationConfig, VehicleConfig
from mlracecar.core.race.events import LapCompleted, OffTrack, RaceEvent, WrongWay
from mlracecar.core.race.rules import (
    SECTORS,
    OffTrackPolicy,
    RaceRules,
    RaceSettings,
    RaceState,
)
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.core.world import World
from mlracecar.io.track_file import read_track_file

GP_CIRCUIT = Path(__file__).parents[4] / "tracks" / "gp-circuit.json"
ANGLES = np.linspace(0, 2 * np.pi, 12, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 12)
LENGTH = CIRCLE.length  # about 377 m
CHECKPOINTS = len(CIRCLE.checkpoints.arc_length)
SPACING = LENGTH / CHECKPOINTS  # about 20 m
RULES = RaceRules(CIRCLE, reach=5.0)
CAR = VehicleConfig().to_params()


def cars_at(arc_length: ArrayLike, offset: ArrayLike = 0.0) -> VehicleState:
    """Cars on the circle, facing the driving direction."""
    pose = CIRCLE.pose_at(np.asarray(arc_length, dtype=np.float64), offset)
    return VehicleState.at_rest(pose.position, pose.heading)


class Drive:
    """Moves cars from spot to spot, one update of 0.1 s per move, recording what happens."""

    def __init__(self, arc_length: ArrayLike) -> None:
        self.spot = cars_at(np.atleast_1d(arc_length))
        self.race: RaceState = RULES.start(self.spot)
        self.time = 0.0
        self.events: list[RaceEvent] = []

    def to(self, arc_length: ArrayLike, offset: float = 0.0) -> None:
        """Move every car to these spots (one for all, or one each)."""
        spots = np.broadcast_to(np.asarray(arc_length, dtype=np.float64), (len(self.spot),))
        spot = cars_at(spots, offset)
        self.race, events = RULES.update(self.race, self.spot, spot, self.time, self.time + 0.1)
        self.spot, self.time = spot, self.time + 0.1
        self.events += events

    def along(self, start: float, stop: float, step: float = 2.0, offset: float = 0.0) -> None:
        """Move every car from ``start`` to ``stop`` along the lap, ``step`` metres per move."""
        for arc_length in np.arange(start, stop, step if stop > start else -step)[1:]:
            self.to(arc_length, offset)
        self.to(stop, offset)

    def laps(self) -> list[LapCompleted]:
        return [event for event in self.events if isinstance(event, LapCompleted)]


def drive_world(track: Track, speed: float) -> tuple[World, CenterlineDriver]:
    world = World(
        track, KinematicBicycle(CAR), SimulationConfig().to_timing(), 1, np.random.default_rng(0)
    )
    return world, CenterlineDriver(track, CAR, speed)


def test_a_new_race_has_no_lap_under_way() -> None:
    race = RULES.start(cars_at([10.0, 200.0], [2.0, -3.0]))

    np.testing.assert_allclose(race.arc_length, [10.0, 200.0], atol=0.02)
    np.testing.assert_allclose(race.offset, [2.0, -3.0], atol=0.01)
    np.testing.assert_allclose(race.heading_error, 0.0, atol=0.01)
    np.testing.assert_array_equal(race.distance, [0.0, 0.0])
    np.testing.assert_array_equal(race.checkpoint, [-1, -1])
    np.testing.assert_array_equal(race.laps, [0, 0])
    assert np.isnan(race.lap_start).all()
    assert race.splits.shape == (2, SECTORS - 1)


# --------------------------------------------------------------------------- #
# Acceptance criteria (#21)
# --------------------------------------------------------------------------- #


def test_reversing_back_and_forth_across_the_finish_line_never_counts_a_lap() -> None:
    drive = Drive(LENGTH - 5.0)
    for _ in range(50):
        drive.to(3.0)  # forwards over the line
        drive.to(LENGTH - 3.0)  # and back again

    assert drive.laps() == []
    assert drive.race.laps[0] == 0
    assert drive.race.checkpoint[0] == -1  # behind the line: no lap under way


def test_backing_all_the_way_round_and_over_the_line_never_counts_a_lap() -> None:
    drive = Drive(LENGTH - 5.0)
    drive.along(LENGTH - 5.0, LENGTH + 30.0)  # start a lap and pass checkpoint 1
    drive.along(LENGTH + 30.0, -LENGTH / 2)  # reverse back over the line and on round
    drive.along(LENGTH / 2, LENGTH + 5.0)  # turn round and drive forwards over the line

    assert drive.laps() == []


def test_skipping_a_checkpoint_invalidates_the_lap() -> None:
    skipped = 5 * SPACING
    drive = Drive(LENGTH - 5.0)
    drive.along(LENGTH - 5.0, LENGTH + skipped - 6.0)
    # Round checkpoint 5 off the road (it is 12 m wide): the line ends at the road's edge.
    drive.along(skipped - 6.0, skipped + 6.0, offset=9.0)
    drive.along(skipped + 6.0, LENGTH + 5.0)

    (lap,) = drive.laps()
    assert not lap.valid
    assert drive.race.laps[0] == 0
    assert np.isnan(drive.race.best_lap[0])


def test_the_scripted_driver_laps_in_the_expected_time() -> None:
    speed = 15.0  # m/s, well within the grip on this 60 m radius circle
    world, driver = drive_world(CIRCLE, speed)
    laps: list[LapCompleted] = []
    while len(laps) < 3:
        events = world.step(driver.act(world.snapshot)).events
        laps += [event for event in events if isinstance(event, LapCompleted)]

    assert all(lap.valid for lap in laps)
    # The first lap starts slowly, from the grid; the others are at full speed.
    for lap in laps[1:]:
        assert lap.time == pytest.approx(LENGTH / speed, rel=0.01)
    assert world.snapshot.race.laps[0] == 3


def test_the_scripted_driver_laps_the_gp_circuit() -> None:
    track = read_track_file(GP_CIRCUIT).to_track()
    world, driver = drive_world(track, 60.0)
    while world.snapshot.race.laps[0] == 0:
        assert world.snapshot.time < 200, "no valid lap after 200 s"
        world.step(driver.act(world.snapshot))

    assert world.snapshot.race.best_lap[0] == pytest.approx(122, abs=5)  # about 100 km/h


# --------------------------------------------------------------------------- #
# Laps and timing
# --------------------------------------------------------------------------- #


def test_the_first_lap_starts_when_the_car_reaches_the_line() -> None:
    drive = Drive(LENGTH - 30.0)
    drive.along(LENGTH - 30.0, LENGTH - 1.0)
    assert np.isnan(drive.race.lap_start[0])

    drive.to(1.0)  # over the line halfway through this 0.1 s update

    assert drive.race.checkpoint[0] == 0
    assert drive.race.lap_start[0] == pytest.approx(drive.time - 0.05)


def test_a_clean_lap_counts_with_its_time_and_sectors() -> None:
    drive = Drive(LENGTH - 1.0)
    drive.along(LENGTH - 1.0, 2 * LENGTH + 1.0, step=1.0)  # 1 m per 0.1 s: 10 m/s

    (lap,) = drive.laps()
    assert lap.valid
    assert lap.time == pytest.approx(LENGTH / 10.0, abs=1e-3)
    assert lap.at == drive.race.lap_start[0]  # the next lap starts as this one ends
    assert sum(lap.sectors) == pytest.approx(lap.time)
    starts = [round(sector * CHECKPOINTS / SECTORS) for sector in range(SECTORS + 1)]
    expected = [(end - begin) * SPACING / 10.0 for begin, end in itertools.pairwise(starts)]
    assert lap.sectors == pytest.approx(expected, abs=1e-3)
    assert drive.race.laps[0] == 1
    assert drive.race.last_lap[0] == drive.race.best_lap[0] == lap.time


def test_the_best_lap_is_the_fastest_valid_one() -> None:
    drive = Drive(LENGTH - 1.0)
    for leg, step in enumerate((2.0, 4.0, 3.0, 1.0)):  # 20, 40, 30, then 10 m/s
        start = LENGTH - 1.0 + leg * LENGTH
        drive.along(start, start + LENGTH, step=step)

    times = [lap.time for lap in drive.laps()]
    # Within 2%: the speed changes between legs fall inside the laps.
    assert times == pytest.approx([LENGTH / 20, LENGTH / 40, LENGTH / 30], rel=0.02)
    assert drive.race.best_lap[0] == times[1]
    assert drive.race.last_lap[0] == times[2]


def test_backing_over_checkpoints_undoes_them_and_the_lap_still_counts() -> None:
    drive = Drive(LENGTH - 5.0)
    drive.along(LENGTH - 5.0, LENGTH + 3 * SPACING + 5.0)
    drive.along(3 * SPACING + 5.0, 2 * SPACING - 5.0)  # back over checkpoints 3 and 2
    assert drive.race.checkpoint[0] == 1
    drive.along(2 * SPACING - 5.0, LENGTH + 5.0)

    (lap,) = drive.laps()
    assert lap.valid


def test_backing_over_a_sector_line_clears_its_split_until_the_car_crosses_it_again() -> None:
    sector_line = round(CHECKPOINTS / SECTORS) * SPACING  # where sector 2 begins
    drive = Drive(LENGTH - 5.0)
    drive.along(LENGTH - 5.0, LENGTH + sector_line + 5.0)
    assert not np.isnan(drive.race.splits[0, 0])

    drive.along(sector_line + 5.0, sector_line - 5.0)
    assert np.isnan(drive.race.splits[0, 0])
    drive.along(sector_line - 5.0, LENGTH + 5.0)

    (lap,) = drive.laps()
    assert lap.valid
    # Sector 1 ends at the second crossing: it includes the 10 m back and 10 m forward again,
    # a second at 20 m/s on top of the straight run to the line.
    assert lap.sectors[0] > sector_line / 20.0 + 0.9
    assert sum(lap.sectors) == pytest.approx(lap.time)


def test_distance_counts_forwards_and_backwards() -> None:
    drive = Drive(10.0)
    drive.along(10.0, LENGTH + 20.0)
    assert drive.race.distance[0] == pytest.approx(LENGTH + 10.0, abs=0.02)

    drive.along(20.0, -30.0)

    assert drive.race.distance[0] == pytest.approx(LENGTH - 40.0, abs=0.02)


def test_a_car_facing_backwards_has_a_heading_error_of_half_a_turn() -> None:
    cars = cars_at([50.0])
    backwards = VehicleState.at_rest(cars.position, cars.yaw + np.pi)

    assert abs(RULES.start(backwards).heading_error[0]) == pytest.approx(np.pi, abs=0.01)


def test_events_come_in_the_order_they_happened() -> None:
    drive = Drive([LENGTH - 30.0, LENGTH - 30.0])
    drive.along(LENGTH - 30.0, 2 * LENGTH - 30.0)  # once round: both laps under way
    drive.to([LENGTH - 3.0, LENGTH - 1.0])
    drive.to([LENGTH + 1.0, LENGTH + 3.0])  # car 1 is over the line first

    first, second = drive.laps()
    assert (first.car, second.car) == (1, 0)
    assert first.at < second.at


# --------------------------------------------------------------------------- #
# Off track and wrong way
# --------------------------------------------------------------------------- #


def test_a_car_is_off_track_while_its_centre_is_off_the_road() -> None:
    drive = Drive(50.0)
    drive.to(52.0, offset=5.9)  # the road is 12 m wide
    assert not drive.race.off_track[0]

    drive.to(54.0, offset=6.1)
    drive.to(56.0, offset=8.0)
    assert drive.race.off_track[0]
    drive.to(58.0, offset=-2.0)
    drive.to(60.0, offset=-7.0)  # off the other side

    excursions = [event for event in drive.events if isinstance(event, OffTrack)]
    assert [(event.car, round(event.arc_length)) for event in excursions] == [(0, 54), (0, 60)]
    assert excursions[0].at == pytest.approx(0.2)


def test_a_car_starting_off_the_road_is_off_track_from_the_start() -> None:
    assert RULES.start(cars_at([50.0, 50.0], [0.0, 10.0])).off_track.tolist() == [False, True]


def test_driving_backwards_along_the_track_is_the_wrong_way() -> None:
    drive = Drive(100.0)
    drive.to(99.95)  # 0.5 m/s backwards: creeping, not driving the wrong way
    assert not drive.race.wrong_way[0]

    drive.along(99.95, 80.0)
    assert drive.race.wrong_way[0]
    drive.along(80.0, 90.0)
    assert not drive.race.wrong_way[0]

    (wrong_way,) = [event for event in drive.events if isinstance(event, WrongWay)]
    assert wrong_way.car == 0
    assert wrong_way.arc_length == pytest.approx(97.95, abs=0.01)


def test_facing_backwards_without_moving_is_not_the_wrong_way() -> None:
    cars = cars_at([50.0])
    backwards = VehicleState.at_rest(cars.position, cars.yaw + np.pi)
    race = RULES.start(backwards)

    race, events = RULES.update(race, backwards, backwards, 0.0, 0.1)

    assert not race.wrong_way[0]
    assert events == ()


def moving(arc_length: float, offset: float, speed: float) -> VehicleState:
    cars = cars_at([arc_length], offset)
    return VehicleState(**{**vars(cars), "vx": np.array([speed]), "yaw_rate": np.array([0.1])})


def off_and_on(policy: OffTrackPolicy) -> tuple[VehicleState, RaceState]:
    """Two cars at 20 m/s, the first off the road, after `enforce`."""
    rules = RaceRules(CIRCLE, reach=5.0, settings=RaceSettings(off_track=policy))
    before = VehicleState(
        **{
            name: np.concatenate([getattr(moving(50.0, 8.0, 20.0), name),
                                  getattr(moving(50.0, 2.0, 20.0), name)])
            for name in vars(moving(0.0, 0.0, 0.0))
        }
    )  # fmt: skip
    race, _ = rules.update(rules.start(before), before, before, 0.0, 0.1)
    return rules.enforce(before, race, 0.1)


def test_with_no_off_track_policy_nothing_happens() -> None:
    cars, race = off_and_on(OffTrackPolicy.NONE)

    np.testing.assert_array_equal(cars.speed, [20.0, 20.0])
    assert race.off_track.tolist() == [True, False]
    assert not race.out.any()


def test_the_grass_slows_a_car_down() -> None:
    cars, race = off_and_on(OffTrackPolicy.SLOWDOWN)

    np.testing.assert_allclose(cars.speed, [20.0 - RaceSettings().grass_slowdown * 0.1, 20.0])
    np.testing.assert_allclose(cars.yaw_rate, [0.1 * cars.speed[0] / 20.0, 0.1])
    assert race.off_track[0]


def test_the_grass_stops_a_slow_car_without_reversing_it() -> None:
    rules = RaceRules(CIRCLE, reach=5.0)
    crawling = moving(50.0, 8.0, 0.2)
    race, _ = rules.update(rules.start(crawling), crawling, crawling, 0.0, 0.1)

    cars, _ = rules.enforce(crawling, race, 0.1)

    assert cars.speed[0] == 0.0


def test_a_reset_puts_the_car_back_in_the_middle_of_the_road_at_rest() -> None:
    cars, race = off_and_on(OffTrackPolicy.RESET)

    # Within a few centimetres: from 8 m off the middle of a bend, finding the spot along the
    # road from the 0.5 m centerline pieces is off by up to 8 · (1/60) · 0.5 / 2 = 3 cm.
    middle = CIRCLE.pose_at([50.0])
    np.testing.assert_allclose(cars.position[0], middle.position[0], atol=0.05)
    assert cars.speed[0] == 0.0
    assert cars.speed[1] == 20.0
    assert race.offset[0] == race.heading_error[0] == 0.0
    assert not race.off_track.any()  # back on the road


def test_leaving_the_road_ends_the_run_with_terminate() -> None:
    cars, race = off_and_on(OffTrackPolicy.TERMINATE)

    assert race.out.tolist() == [True, False]
    np.testing.assert_array_equal(cars.speed, [0.0, 20.0])
