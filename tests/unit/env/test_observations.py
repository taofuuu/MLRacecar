"""Tests for mlracecar.env.observations: what the AI sees, and the spec that describes it."""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import ArrayLike

from mlracecar.config.models import OBSERVATION_INPUTS, ObservationConfig, VehicleConfig
from mlracecar.core.race.rules import RaceRules
from mlracecar.core.sensors import RaySettings
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.env.observations import (
    ObservationBuilder,
    ObservationSpec,
    observation_differences,
)
from mlracecar.io.track_file import read_track_file
from strategies import cars_and_actions

CAR = VehicleConfig().to_params()
TECHNICAL = read_track_file(Path(__file__).parents[3] / "tracks" / "technical.json").to_track()


def circle(radius: float, *, clockwise: bool = False) -> Track:
    """A round track 12 m wide; counter-clockwise unless asked otherwise."""
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    if clockwise:
        angles = -angles
    return Track.build(radius * np.column_stack([np.cos(angles), np.sin(angles)]), [12.0] * 48)


CIRCLE = circle(60.0)


def snapshot(track: Track, cars: VehicleState) -> Snapshot:
    """The cars as they are, each with its place along the lap worked out."""
    return Snapshot(0, 0.0, cars, RaceRules(track, reach=1.0).start(cars), ())


def one_car(**state: float) -> VehicleState:
    """A car at (60, 0) on `CIRCLE`, driving along the road, unless told otherwise."""
    values = {
        "x": 60.0,
        "y": 0.0,
        "yaw": math.pi / 2,
        "vx": 0.0,
        "vy": 0.0,
        "yaw_rate": 0.0,
        "steer": 0.0,
    }
    values.update(state)
    return VehicleState(**{name: np.array([value]) for name, value in values.items()})


def only(*names: str, **settings: float) -> ObservationConfig:
    """The observation settings with only these inputs on."""
    return ObservationConfig.model_validate(
        {name: name in names for name in OBSERVATION_INPUTS} | settings
    )


def observe(
    config: ObservationConfig,
    cars: VehicleState,
    actions: ArrayLike | None = None,
    track: Track = CIRCLE,
) -> np.ndarray:
    builder = ObservationBuilder(track, CAR, config)
    return builder.build(
        snapshot(track, cars), np.zeros((len(cars), 2)) if actions is None else actions
    )


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #


def test_every_input_is_in_by_default_in_a_fixed_order() -> None:
    spec = ObservationBuilder(CIRCLE, CAR, ObservationConfig()).spec

    assert [feature.name for feature in spec.features] == list(OBSERVATION_INPUTS)
    assert spec.size == 15 + 1 + 2 + 1 + 1 + 1 + 2 + 8
    assert spec.labels[:2] == ("rays[0]", "rays[1]")
    assert spec.labels[15:24] == (
        "speed",
        "heading.sin",
        "heading.cos",
        "offset",
        "yaw_rate",
        "steering",
        "previous_action.steer",
        "previous_action.pedal",
        "curvature[0]",
    )
    assert spec.low.shape == spec.high.shape == (spec.size,)
    assert spec.low.dtype == np.float32
    assert np.all(spec.low < spec.high)


def test_the_default_digest_is_pinned() -> None:
    # A trained model only fits observations with its digest. If this fails, the default
    # observation changed: say so in the pull request, because earlier models won't load.
    assert ObservationBuilder(CIRCLE, CAR, ObservationConfig()).spec.digest == DEFAULT_DIGEST


DEFAULT_DIGEST = "41c17b29c79e18ca17a22a897646bbc965789e49582afbebffc4f01457261da2"


def test_the_same_settings_give_the_same_digest() -> None:
    first = ObservationBuilder(CIRCLE, CAR, ObservationConfig()).spec
    second = ObservationBuilder(
        TECHNICAL, CAR, ObservationConfig()
    ).spec  # the track doesn't matter

    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert int(first.digest, 16) >= 0  # hex digits


@pytest.mark.parametrize("name", OBSERVATION_INPUTS)
def test_turning_any_input_off_changes_the_digest(name: str) -> None:
    default = ObservationBuilder(CIRCLE, CAR, ObservationConfig()).spec
    without = ObservationBuilder(CIRCLE, CAR, ObservationConfig.model_validate({name: False})).spec

    assert name not in [feature.name for feature in without.features]
    assert without.digest != default.digest


@pytest.mark.parametrize(
    ("config", "rays"),
    [
        (ObservationConfig(lookahead=200.0), None),
        (ObservationConfig(lookahead_points=6), None),
        (ObservationConfig(), RaySettings(count=19)),
        (ObservationConfig(), RaySettings(field_of_view=math.pi / 2)),
        (ObservationConfig(), RaySettings(max_range=50.0)),
    ],
)
def test_changing_how_an_input_is_measured_changes_the_digest(
    config: ObservationConfig, rays: RaySettings | None
) -> None:
    default = ObservationBuilder(CIRCLE, CAR, ObservationConfig()).spec

    assert ObservationBuilder(CIRCLE, CAR, config, rays).spec.digest != default.digest


def test_an_empty_spec_has_nothing_in_it() -> None:
    assert ObservationSpec(()).size == 0
    assert ObservationSpec(()).labels == ()


# --------------------------------------------------------------------------- #
# Each input
# --------------------------------------------------------------------------- #


def test_observations_are_float32_rows_one_per_car() -> None:
    cars = VehicleState.at_rest(
        np.array([[60.0, 0.0], [0.0, 60.0]]), np.array([math.pi / 2, math.pi])
    )

    observations = observe(ObservationConfig(), cars)

    assert observations.shape == (2, 31)
    assert observations.dtype == np.float32


def test_rays_are_fractions_of_their_range_and_kept_for_drawing() -> None:
    builder = ObservationBuilder(CIRCLE, CAR, only("rays"))

    observations = builder.build(snapshot(CIRCLE, one_car()), np.zeros((1, 2)))

    assert builder.readings is not None
    np.testing.assert_array_equal(observations, builder.readings.normalized.astype(np.float32))
    assert observations[0, 0] == pytest.approx(0.06, abs=1e-3)  # 6 m to the right edge, of 100 m


def test_without_rays_there_is_no_sensor() -> None:
    builder = ObservationBuilder(CIRCLE, CAR, only("speed"))

    builder.build(snapshot(CIRCLE, one_car()), np.zeros((1, 2)))

    assert builder.sensor is None
    assert builder.readings is None


def test_speed_counts_100_metres_a_second_as_1() -> None:
    assert observe(only("speed"), one_car(vx=50.0)).tolist() == [[0.5]]


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (math.pi / 2, [0.0, 1.0]),  # along the road
        (math.pi, [1.0, 0.0]),  # turned a quarter left, across the road
        (-math.pi / 2, [0.0, -1.0]),  # backwards
    ],
)
def test_heading_is_the_sine_and_cosine_of_the_angle_to_the_road(
    yaw: float, expected: list[float]
) -> None:
    np.testing.assert_allclose(observe(only("heading"), one_car(yaw=yaw)), [expected], atol=1e-3)


@pytest.mark.parametrize(("x", "expected"), [(60.0, 0.0), (57.0, 0.5), (66.0, -1.0), (80.0, -2.0)])
def test_offset_is_a_share_of_half_the_road_left_positive(x: float, expected: float) -> None:
    # Counter-clockwise, the inside of the circle is on the left. Far out on the grass the
    # value stops at -2.
    np.testing.assert_allclose(observe(only("offset"), one_car(x=x)), [[expected]], atol=1e-3)


def test_yaw_rate_counts_2_radians_a_second_as_1() -> None:
    np.testing.assert_allclose(observe(only("yaw_rate"), one_car(yaw_rate=-1.0)), [[-0.5]])


def test_steering_is_a_share_of_full_lock() -> None:
    np.testing.assert_allclose(
        observe(only("steering"), one_car(steer=CAR.max_steer / 4)), [[0.25]]
    )


def test_the_previous_action_is_passed_on_clipped() -> None:
    observations = observe(only("previous_action"), one_car(), [[0.5, -1.5]])

    np.testing.assert_allclose(observations, [[0.5, -1.0]])


@pytest.mark.parametrize("actions", [np.zeros((2, 2)), [[math.nan, 0.0]]])
def test_impossible_previous_actions_are_refused(actions: ArrayLike) -> None:
    with pytest.raises(ValueError, match="actions"):
        observe(ObservationConfig(), one_car(), actions)


def test_curvature_on_a_circle_is_one_over_its_radius_in_every_stretch() -> None:
    observations = observe(only("curvature"), one_car())

    np.testing.assert_allclose(observations, [[10 / 60] * 8], atol=1e-4)


def test_curvature_is_negative_for_right_hand_bends() -> None:
    clockwise = circle(60.0, clockwise=True)
    car = one_car(yaw=-math.pi / 2)

    observations = observe(only("curvature"), car, track=clockwise)

    np.testing.assert_allclose(observations, [[-10 / 60] * 8], atol=1e-4)


def test_curvature_carries_on_past_the_start_line() -> None:
    # Just before the line, most of the 150 m ahead is on the next lap.
    near_the_end = one_car(x=60 * math.cos(-0.05), y=60 * math.sin(-0.05), yaw=math.pi / 2 - 0.05)

    np.testing.assert_allclose(observe(only("curvature"), near_the_end), [[10 / 60] * 8], atol=1e-4)


@pytest.mark.parametrize("start", [0.0, 333.3, 702.0, 1050.0])
def test_curvature_is_the_tracks_own_curvature_averaged_over_each_stretch(start: float) -> None:
    builder = ObservationBuilder(TECHNICAL, CAR, only("curvature"))
    pose = TECHNICAL.pose_at(start)

    bends = builder.build(
        snapshot(TECHNICAL, VehicleState.at_rest(pose.position, pose.heading)), np.zeros((1, 2))
    )

    line = TECHNICAL.centerline
    expected = []
    for stretch in range(8):
        ahead = start + np.linspace(stretch * 18.75, (stretch + 1) * 18.75, 200)
        curvature = np.interp(ahead, line.arc_length, line.curvature, period=line.length)
        expected.append(10 * np.trapezoid(curvature, ahead) / 18.75)
    np.testing.assert_allclose(bends[0], expected, atol=0.01)
    assert np.abs(bends).max() > 0.1  # a real bend is in view, not just a straight


# --------------------------------------------------------------------------- #
# Any state at all
# --------------------------------------------------------------------------- #


@given(cars_and_actions(max_steer=CAR.max_steer), st.sampled_from([CIRCLE, TECHNICAL]))
def test_every_value_is_finite_and_within_its_bounds(
    cars_actions: tuple[VehicleState, np.ndarray], track: Track
) -> None:
    # Cars anywhere within a kilometre, far off the road included, moving at up to 120 m/s.
    cars, actions = cars_actions
    builder = ObservationBuilder(track, CAR, ObservationConfig())

    observations = builder.build(snapshot(track, cars), actions)

    assert np.isfinite(observations).all()
    assert (observations >= builder.spec.low).all()
    assert (observations <= builder.spec.high).all()


@given(
    st.lists(
        st.booleans(), min_size=len(OBSERVATION_INPUTS), max_size=len(OBSERVATION_INPUTS)
    ).filter(any),
    st.integers(1, 12),
)
def test_the_spec_says_how_many_values_there_are_whatever_is_chosen(
    switches: list[bool], points: int
) -> None:
    config = ObservationConfig.model_validate(
        dict(zip(OBSERVATION_INPUTS, switches, strict=True)) | {"lookahead_points": points}
    )
    builder = ObservationBuilder(CIRCLE, CAR, config)

    observations = builder.build(snapshot(CIRCLE, one_car()), np.zeros((1, 2)))

    assert observations.shape == (1, builder.spec.size)
    assert len(builder.spec.labels) == builder.spec.size


# --------------------------------------------------------------------------- #
# Saying how two observations differ
# --------------------------------------------------------------------------- #


def description(config: ObservationConfig, rays: RaySettings | None = None) -> dict[str, Any]:
    return ObservationBuilder(CIRCLE, CAR, config, rays).spec.description()


def test_the_digest_is_made_from_the_description() -> None:
    spec = ObservationBuilder(CIRCLE, CAR, ObservationConfig()).spec
    text = json.dumps(spec.description(), sort_keys=True)

    assert hashlib.sha256(text.encode()).hexdigest() == spec.digest == DEFAULT_DIGEST


def test_the_same_observations_have_no_differences() -> None:
    assert (
        observation_differences(description(ObservationConfig()), description(ObservationConfig()))
        == []
    )


@pytest.mark.parametrize(
    ("saved", "current", "expected"),
    [
        (
            ObservationConfig(),
            ObservationConfig(curvature=False),
            ["curvature: in the model, but turned off here"],
        ),
        (
            ObservationConfig(speed=False),
            ObservationConfig(),
            ["speed: turned on here, but not in the model"],
        ),
        (
            ObservationConfig(),
            ObservationConfig(lookahead=200.0),
            ["curvature: stretch is 25.0 here, 18.75 in the model"],
        ),
        (
            ObservationConfig(),
            ObservationConfig(lookahead_points=4, lookahead=75.0),
            ["curvature: 4 values here, 8 in the model"],
        ),
    ],
)
def test_each_difference_is_named(
    saved: ObservationConfig, current: ObservationConfig, expected: list[str]
) -> None:
    assert observation_differences(description(saved), description(current)) == expected


def test_different_rays_are_named() -> None:
    differences = observation_differences(
        description(ObservationConfig()), description(ObservationConfig(), RaySettings(count=19))
    )

    assert differences == [
        "rays: 19 values here, 15 in the model",
        "rays: count is 19 here, 15 in the model",
    ]


def test_a_new_format_order_or_bounds_are_named() -> None:
    saved = description(only("speed", "offset"))
    current = copy.deepcopy(saved)
    current["version"] = 2
    current["features"].reverse()
    current["features"][0]["high"] = 3.0  # offset

    assert observation_differences(saved, current) == [
        "observation format: version 2 here, 1 in the model",
        "offset: bounds -2.0..3.0 here, -2.0..2.0 in the model",
        "the inputs are in a different order",
    ]
