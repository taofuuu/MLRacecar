"""The settings: every value MLRacecar reads from settings files, with its default (ADR-0008).

Each section converts to the plain frozen dataclass the simulation uses, so `mlracecar.core`
never sees pydantic. Settings files use units that are easy to picture, such as angles in degrees
and power in kilowatts; the conversion turns them into SI units.

Every setting's docstring becomes its comment in the YAML written by
`mlracecar.config.files.format_config`, so keep them to one short line.
"""

import math
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from mlracecar.core.race.rules import OffTrackPolicy, RaceSettings
from mlracecar.core.sensors import RaySettings
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.core.world import Timing

_SETTINGS = ConfigDict(
    extra="forbid",  # a misspelled setting is an error, not silently ignored
    frozen=True,
    strict=True,  # no guessing: "12" is not a number, 1.5 is not a whole number
    allow_inf_nan=False,
    validate_default=True,  # e.g. the default wheelbase must still fit a shorter car
    use_attribute_docstrings=True,
)

Positive = Annotated[float, Field(gt=0)]
NonNegative = Annotated[float, Field(ge=0)]

OBSERVATION_INPUTS = (
    "rays",
    "speed",
    "heading",
    "offset",
    "yaw_rate",
    "steering",
    "previous_action",
    "curvature",
)
"""The inputs an observation can have, in the order they appear in it."""


class VehicleConfig(BaseModel):
    """The car. The defaults describe a light race car: quick, and very grippy and hard-braking."""

    model_config = _SETTINGS

    length: Positive = 4.5
    """Length of the body, in metres."""
    width: Positive = 2.0
    """Width of the body, in metres."""
    wheelbase: Positive = 2.7
    """Front axle to rear axle, in metres."""
    mass: Positive = 1300.0
    """The car's mass, in kilograms."""
    max_steer: Annotated[float, Field(gt=0, lt=90)] = 30.0
    """How far the front wheels turn at full lock, in degrees."""
    steer_rate: Positive = 180.0
    """How fast the front wheels can turn, in degrees per second."""
    grip: Positive = 2.0
    """Tyre grip: the hardest the car can corner, in g (gravity = 1)."""
    max_drive_force: Positive = 12000.0
    """The engine's push at low speed, in newtons."""
    max_power: Positive = 250.0
    """Engine power in kilowatts; it limits the push at high speed."""
    max_brake_force: Positive = 37000.0
    """The brakes' strongest push, in newtons."""
    drag_coefficient: NonNegative = 0.32
    """Air resistance of the car's shape (Cd, no unit)."""
    frontal_area: Positive = 2.0
    """Size of the car seen from the front, in square metres."""
    rolling_resistance: NonNegative = 0.015
    """Tyre rolling resistance, as a fraction of the car's weight."""

    @field_validator("wheelbase")
    @classmethod
    def _shorter_than_the_car(cls, wheelbase: float, info: ValidationInfo) -> float:
        length = info.data.get("length")  # missing if the length itself was invalid
        if length is not None and wheelbase >= length:
            raise PydanticCustomError(
                "wheelbase_too_long",
                "must be shorter than the car's length ({length} m)",
                {"length": length},
            )
        return wheelbase

    def to_params(self) -> VehicleParams:
        """The car in SI units, for the simulation."""
        return VehicleParams(
            length=self.length,
            width=self.width,
            wheelbase=self.wheelbase,
            mass=self.mass,
            max_steer=math.radians(self.max_steer),
            steer_rate=math.radians(self.steer_rate),
            grip=self.grip,
            max_drive_force=self.max_drive_force,
            max_power=self.max_power * 1000,
            max_brake_force=self.max_brake_force,
            drag_coefficient=self.drag_coefficient,
            frontal_area=self.frontal_area,
            rolling_resistance=self.rolling_resistance,
        )


class SimulationConfig(BaseModel):
    """How simulated time moves forward."""

    model_config = _SETTINGS

    physics_hz: Annotated[int, Field(ge=1)] = 120
    """Physics steps per simulated second."""
    action_repeat: Annotated[int, Field(ge=1)] = 6
    """Physics steps per driver decision (120 / 6 = 20 decisions a second)."""

    def to_timing(self) -> Timing:
        """The timing, for the simulation."""
        return Timing(physics_hz=self.physics_hz, action_repeat=self.action_repeat)


class RaceConfig(BaseModel):
    """The race rules."""

    model_config = _SETTINGS

    off_track: Literal["none", "slowdown", "reset", "terminate"] = "slowdown"
    """When a car's centre leaves the road: none, slowdown, reset, or terminate."""
    grass_slowdown: Positive = 6.0
    """How hard grass slows a car, in m/s per second (off_track: slowdown)."""

    def to_settings(self) -> RaceSettings:
        """The settings, for the race rules."""
        return RaceSettings(
            off_track=OffTrackPolicy(self.off_track), grass_slowdown=self.grass_slowdown
        )


class SensorConfig(BaseModel):
    """The car's distance sensors: rays that measure how far away the road's edges are."""

    model_config = _SETTINGS

    rays: Annotated[int, Field(ge=1)] = 15
    """How many rays fan out from the car."""
    field_of_view: Annotated[float, Field(gt=0, le=360)] = 180.0
    """The angle they fan across, in degrees, centred on where the car points."""
    range: Positive = 100.0
    """How far they reach, in metres."""

    def to_settings(self) -> RaySettings:
        """The settings, for the sensors."""
        return RaySettings(
            count=self.rays,
            field_of_view=math.radians(self.field_of_view),
            max_range=self.range,
        )


class ObservationConfig(BaseModel):
    """What the AI sees: which inputs go into its observation, in this order."""

    model_config = _SETTINGS

    rays: bool = True
    """The distance rays, each as a fraction of their range."""
    speed: bool = True
    """The car's forward speed."""
    heading: bool = True
    """Which way the car points compared with the road (2 values)."""
    offset: bool = True
    """Distance from the middle of the road, as a share of half its width."""
    yaw_rate: bool = True
    """How fast the car is turning."""
    steering: bool = True
    """Where the front wheels point now, as a share of full lock."""
    previous_action: bool = True
    """The steering and pedal it chose last time (2 values)."""
    curvature: bool = True
    """How much the road bends in each stretch ahead (one value per stretch)."""
    lookahead: Positive = 150.0
    """How far ahead the bends are measured, in metres."""
    lookahead_points: Annotated[int, Field(ge=1)] = 8
    """How many equal stretches that distance is split into."""

    @model_validator(mode="after")
    def _sees_something(self) -> Self:
        if not any(getattr(self, name) for name in OBSERVATION_INPUTS):
            raise PydanticCustomError("no_inputs", "turn on at least one input")
        return self


class RacecarConfig(BaseModel):
    """Every setting, in sections. `mlracecar.config.files.load_config` builds one from files."""

    model_config = _SETTINGS

    vehicle: VehicleConfig = Field(default_factory=VehicleConfig)
    """The car: its size, steering, engine and brakes, and what slows it down."""
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    """How simulated time moves forward."""
    race: RaceConfig = Field(default_factory=RaceConfig)
    """The race rules: what happens when a car leaves the road."""
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    """The car's distance sensors: how many rays, how wide they fan, and how far they reach."""
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    """What the AI sees: each input can be turned off with false."""
