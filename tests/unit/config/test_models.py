"""Tests for mlracecar.config.models: defaults, checks, and conversion to the core dataclasses."""

import dataclasses
import math

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from mlracecar.config.models import RacecarConfig, SimulationConfig, VehicleConfig
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.core.world import Timing

SECTIONS: list[type[BaseModel]] = [VehicleConfig, SimulationConfig]


def test_every_setting_has_a_default() -> None:
    assert RacecarConfig.model_validate({}) == RacecarConfig()


@pytest.mark.parametrize("model", [RacecarConfig, *SECTIONS])
def test_every_setting_has_a_one_line_explanation(model: type[BaseModel]) -> None:
    # The explanations become comments in the settings files MLRacecar writes.
    for name, field in model.model_fields.items():
        assert field.description, f"{model.__name__}.{name} needs a docstring"
        assert "\n" not in field.description.strip(), f"{model.__name__}.{name}: keep it to a line"


def test_settings_cannot_be_changed_after_loading() -> None:
    config = RacecarConfig()

    with pytest.raises(ValidationError, match="frozen"):
        config.vehicle.mass = 900.0


def test_the_default_car_is_a_sporty_road_car() -> None:
    car = VehicleConfig().to_params()
    air_density, gravity = 1.225, 9.81
    drag = 0.5 * air_density * car.drag_coefficient * car.frontal_area
    rolling = car.rolling_resistance * car.mass * gravity

    # Flat out, the engine's power just covers the air and the tyres: P = v (drag v² + rolling).
    roots = np.roots([drag, 0, rolling, -car.max_power])
    top_speed = max(root.real for root in roots if abs(root.imag) < 1e-9)

    assert 270 < top_speed * 3.6 < 290  # km/h


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def test_numbers_must_be_numbers_and_whole_numbers_whole() -> None:
    with pytest.raises(ValidationError) as caught:
        RacecarConfig.model_validate(
            {"vehicle": {"mass": "1300"}, "simulation": {"physics_hz": 120.0}}
        )

    assert [error["loc"] for error in caught.value.errors()] == [
        ("vehicle", "mass"),
        ("simulation", "physics_hz"),
    ]


def test_whole_numbers_are_fine_where_decimals_are_expected() -> None:
    assert VehicleConfig.model_validate({"mass": 1500}).mass == 1500.0


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("mass", 0),
        ("mass", math.inf),
        ("mass", math.nan),
        ("max_steer", 90),
        ("drag_coefficient", -0.1),
        ("rolling_resistance", -0.1),
        ("wheelbase", 4.5),  # as long as the car
    ],
)
def test_impossible_cars_are_rejected(setting: str, value: float) -> None:
    with pytest.raises(ValidationError) as caught:
        VehicleConfig.model_validate({setting: value})

    assert caught.value.errors()[0]["loc"] == (setting,)


def test_no_drag_or_rolling_resistance_is_allowed() -> None:
    car = VehicleConfig(drag_coefficient=0.0, rolling_resistance=0.0)

    assert car.drag_coefficient == car.rolling_resistance == 0.0


def test_the_wheelbase_is_checked_against_the_length_given() -> None:
    assert VehicleConfig(length=2.5, wheelbase=2.0).wheelbase == 2.0
    with pytest.raises(ValidationError, match=r"shorter than the car's length \(2.5 m\)"):
        VehicleConfig(length=2.5)


def test_an_invalid_length_is_reported_once() -> None:
    with pytest.raises(ValidationError) as caught:
        VehicleConfig.model_validate({"length": -1})

    assert [error["loc"] for error in caught.value.errors()] == [("length",)]


@pytest.mark.parametrize("setting", ["physics_hz", "action_repeat"])
def test_time_needs_at_least_one_step(setting: str) -> None:
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate({setting: 0})


# --------------------------------------------------------------------------- #
# Conversion to the core
# --------------------------------------------------------------------------- #


def test_every_car_setting_reaches_the_simulation() -> None:
    assert {field.name for field in dataclasses.fields(VehicleParams)} == set(
        VehicleConfig.model_fields
    )


def test_car_settings_convert_to_si_units() -> None:
    config = VehicleConfig(max_steer=30.0, steer_rate=90.0, max_power=200.0)

    params = config.to_params()

    assert params.max_steer == pytest.approx(math.pi / 6)  # degrees -> radians
    assert params.steer_rate == pytest.approx(math.pi / 2)
    assert params.max_power == 200_000.0  # kilowatts -> watts
    unchanged = set(VehicleConfig.model_fields) - {"max_steer", "steer_rate", "max_power"}
    assert {name: getattr(params, name) for name in unchanged} == {
        name: getattr(config, name) for name in unchanged
    }


def test_simulation_settings_convert_to_timing() -> None:
    timing = SimulationConfig(physics_hz=100, action_repeat=4).to_timing()

    assert timing == Timing(physics_hz=100, action_repeat=4)
