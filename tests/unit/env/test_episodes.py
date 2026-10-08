"""Tests for mlracecar.env.episodes: when a run ends, and whether it ended or was stopped."""

from dataclasses import replace

import numpy as np
import pytest

from mlracecar.config.models import EpisodeConfig, SimulationConfig, VehicleConfig
from mlracecar.core.race.events import OffTrack
from mlracecar.core.race.rules import OffTrackPolicy, RaceSettings
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import World
from mlracecar.env.episodes import STUCK_SPEED, EndReason, EpisodeRules

CAR = VehicleConfig().to_params()
TIMING = SimulationConfig().to_timing()
DT = TIMING.decision_dt
ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 48)


def moving(speed: float = 20.0, cars: int = 2) -> Snapshot:
    """Cars on the grid of a round track, all moving at ``speed``."""
    snapshot = World(CIRCLE, KinematicBicycle(CAR), TIMING, cars, np.random.default_rng(0)).snapshot
    return replace(snapshot, cars=replace(snapshot.cars, vx=np.full(cars, speed)))


def with_race(
    snapshot: Snapshot, events: tuple[OffTrack, ...] = (), **race: np.ndarray
) -> Snapshot:
    return replace(snapshot, race=replace(snapshot.race, **race), events=events)


def rules(**settings: object) -> EpisodeRules:
    return EpisodeRules(EpisodeConfig.model_validate(settings), DT, cars=2)


# --------------------------------------------------------------------------- #
# Stopped: the time limit and getting stuck (truncated)
# --------------------------------------------------------------------------- #


def test_sixty_seconds_is_exactly_1200_steps() -> None:
    episodes = rules(time_limit=60.0)
    snapshot = moving()

    for _ in range(1199):
        assert not episodes.check(snapshot).truncated.any()
    endings = episodes.check(snapshot)

    assert episodes.max_steps == 1200
    assert endings.truncated.all()
    assert not endings.terminated.any()
    assert endings.reasons == (EndReason.TIME_LIMIT, EndReason.TIME_LIMIT)


def test_a_time_limit_between_steps_rounds_up() -> None:
    assert rules(time_limit=0.12).max_steps == 3


def test_a_car_slower_than_1_m_s_for_5_seconds_is_stuck() -> None:
    episodes = rules(stuck_time=5.0)
    snapshot = moving()
    snapshot = replace(snapshot, cars=replace(snapshot.cars, vx=np.array([0.5, 20.0])))

    for _ in range(99):
        assert not episodes.check(snapshot).truncated.any()
    endings = episodes.check(snapshot)

    assert endings.truncated.tolist() == [True, False]
    assert endings.reasons == (EndReason.STUCK, None)


def test_moving_again_starts_the_stuck_clock_over() -> None:
    episodes = rules(stuck_time=1.0)  # 20 steps
    stuck = replace(moving(), cars=replace(moving().cars, vx=np.full(2, STUCK_SPEED / 2)))
    for _ in range(19):
        episodes.check(stuck)

    episodes.check(moving())  # one step at speed

    for _ in range(19):
        assert not episodes.check(stuck).truncated.any()
    assert episodes.check(stuck).truncated.all()


# --------------------------------------------------------------------------- #
# Over: leaving the road, or taken out by the race rules (terminated)
# --------------------------------------------------------------------------- #


def test_a_car_off_the_road_ends_its_run() -> None:
    snapshot = with_race(moving(), off_track=np.array([False, True]))

    endings = rules().check(snapshot)

    assert endings.terminated.tolist() == [False, True]
    assert not endings.truncated.any()
    assert endings.reasons == (None, EndReason.OFF_TRACK)


def test_leaving_the_road_ends_the_run_even_if_the_car_was_put_straight_back() -> None:
    # With race.off_track: reset, the car is back on the road by the end of the step, but the
    # event says it left.
    snapshot = with_race(moving(), events=(OffTrack(car=0, arc_length=5.0, at=1.0),))

    assert rules().check(snapshot).terminated.tolist() == [True, False]


def test_leaving_the_road_can_be_allowed() -> None:
    snapshot = with_race(moving(), off_track=np.array([True, True]))

    endings = rules(end_off_track=False).check(snapshot)

    assert not endings.terminated.any()
    assert endings.reasons == (None, None)


def test_a_car_the_race_rules_took_out_is_over_either_way() -> None:
    snapshot = with_race(moving(), out=np.array([True, False]))

    endings = rules(end_off_track=False).check(snapshot)

    assert endings.terminated.tolist() == [True, False]
    assert endings.reasons == (EndReason.OUT, None)


def test_a_run_that_is_over_is_never_also_reported_as_stopped() -> None:
    episodes = rules(time_limit=DT)  # one step
    snapshot = with_race(moving(STUCK_SPEED / 2), off_track=np.array([True, False]))

    endings = episodes.check(snapshot)

    assert endings.terminated.tolist() == [True, False]
    assert endings.truncated.tolist() == [False, True]
    assert endings.reasons == (EndReason.OFF_TRACK, EndReason.TIME_LIMIT)


# --------------------------------------------------------------------------- #
# Starting over
# --------------------------------------------------------------------------- #


def test_starting_again_resets_only_the_cars_asked() -> None:
    episodes = rules()
    for _ in range(10):
        episodes.check(moving(STUCK_SPEED / 2))

    episodes.start(np.array([True, False]))

    assert episodes.steps.tolist() == [0, 10]
    assert episodes.stuck_steps.tolist() == [0, 10]
    episodes.start()
    assert episodes.steps.tolist() == [0, 0]


@pytest.mark.parametrize("policy", list(OffTrackPolicy))
def test_driving_straight_off_a_bend_ends_the_run_whatever_the_race_rule(
    policy: OffTrackPolicy,
) -> None:
    world = World(
        CIRCLE,
        KinematicBicycle(CAR),
        TIMING,
        1,
        np.random.default_rng(0),
        settings=RaceSettings(off_track=policy),
    )
    episodes = EpisodeRules(EpisodeConfig(), DT, cars=1)
    straight = np.array([[0.0, 1.0]])  # straight on, while the road curves away to the left

    for _ in range(200):
        endings = episodes.check(world.step(straight))
        if endings.terminated[0]:
            break

    assert endings.reasons == (EndReason.OFF_TRACK,)
    assert episodes.steps[0] < 200
