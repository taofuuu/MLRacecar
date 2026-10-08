"""What the AI sees: observations built from snapshots (architecture section 4.6).

An observation is one row of numbers per car, the same length every step, scaled to about
-1..1 so that a neural network can learn from it. The `observation` settings
(`mlracecar.config.models.ObservationConfig`) choose which inputs go in; they always appear
in the order of `OBSERVATION_INPUTS`.

Every observation comes with an `ObservationSpec`: each input's name, size, bounds, and the
constants that scale it, and a digest (a fingerprint) of all of that. A trained model is saved
with the digest of the observations it learned from, so that it can't be fed different ones
without anyone noticing (M4).
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlracecar.config.models import OBSERVATION_INPUTS, ObservationConfig
from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.sensors import RayReadings, RaySensor, RaySettings
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.dynamics import checked_actions
from mlracecar.core.vehicle.params import VehicleParams

SPEED_SCALE = 100.0
"""The speed, in m/s, that counts as 1. The default car's top speed is about 84 m/s."""

YAW_RATE_SCALE = 2.0
"""The turning rate, in radians a second, that counts as 1: about the most the default car's
grip allows."""

CURVATURE_SCALE = 10.0
"""The radius, in metres, of a bend that counts as 1. A 12 m hairpin is about 0.8."""

LIMIT = 2.0
"""Scaled inputs are clipped to ``-LIMIT..LIMIT``, so even a car flung far off the road, or a
far faster car, gives bounded numbers."""

SPEC_VERSION = 1
"""Raise it when an input's meaning changes without any setting changing, so the digest does."""

type _Compute = Callable[[Snapshot, FloatArray], FloatArray]


@dataclass(frozen=True)
class Feature:
    """One input: its name, a label for each of its values, their bounds, and the constants
    that shape them."""

    name: str
    labels: tuple[str, ...]
    low: float
    high: float
    constants: tuple[tuple[str, float], ...] = ()

    @property
    def size(self) -> int:
        """How many values it adds to the observation."""
        return len(self.labels)


@dataclass(frozen=True)
class ObservationSpec:
    """What an observation holds, in order."""

    features: tuple[Feature, ...]

    @property
    def size(self) -> int:
        """How many values each car's observation has."""
        return sum(feature.size for feature in self.features)

    @property
    def labels(self) -> tuple[str, ...]:
        """A name for every value, such as ``rays[3]`` or ``heading.cos``, for logs and plots."""
        return tuple(label for feature in self.features for label in feature.labels)

    @property
    def low(self) -> NDArray[np.float32]:
        """The smallest each value can be, shape ``(size,)``."""
        return self._bound(lambda feature: feature.low)

    @property
    def high(self) -> NDArray[np.float32]:
        """The largest each value can be, shape ``(size,)``."""
        return self._bound(lambda feature: feature.high)

    @property
    def digest(self) -> str:
        """A fingerprint of everything that shapes the observation, as 64 hex digits. Two specs
        have the same digest exactly when they describe the same numbers in the same order."""
        text = json.dumps(self.description(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def description(self) -> dict[str, Any]:
        """Everything that shapes the observation, as plain JSON-ready data: what the digest is
        made from, and what a model card keeps to say how two observations differ."""
        return {
            "version": SPEC_VERSION,
            "features": [
                {
                    "name": feature.name,
                    "labels": list(feature.labels),
                    "low": feature.low,
                    "high": feature.high,
                    "constants": dict(feature.constants),
                }
                for feature in self.features
            ],
        }

    def _bound(self, bound: Callable[[Feature], float]) -> NDArray[np.float32]:
        return np.concatenate(
            [np.full(feature.size, bound(feature), dtype=np.float32) for feature in self.features]
        )


def observation_differences(saved: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """How two observations differ, in plain words, one line per difference.

    Args:
        saved: The description (`ObservationSpec.description`) a model was trained with.
        current: The description of the observations it would be given now.
    """
    problems: list[str] = []
    if saved.get("version") != current.get("version"):
        problems.append(
            f"observation format: version {current.get('version')} here, "
            f"{saved.get('version')} in the model"
        )
    before = {feature["name"]: feature for feature in saved.get("features", [])}
    now = {feature["name"]: feature for feature in current.get("features", [])}
    problems += [f"{name}: in the model, but turned off here" for name in before if name not in now]
    problems += [
        f"{name}: turned on here, but not in the model" for name in now if name not in before
    ]
    for name in (name for name in now if name in before):
        old, new = before[name], now[name]
        if len(old["labels"]) != len(new["labels"]):
            problems.append(
                f"{name}: {len(new['labels'])} values here, {len(old['labels'])} in the model"
            )
        for key in sorted(set(old["constants"]) | set(new["constants"])):
            here, there = new["constants"].get(key), old["constants"].get(key)
            if here != there:
                problems.append(f"{name}: {key} is {here} here, {there} in the model")
        if (old["low"], old["high"]) != (new["low"], new["high"]):
            problems.append(
                f"{name}: bounds {new['low']}..{new['high']} here, "
                f"{old['low']}..{old['high']} in the model"
            )
    if [name for name in before if name in now] != [name for name in now if name in before]:
        problems.append("the inputs are in a different order")
    return problems


class ObservationBuilder:
    """Builds every car's observation from a snapshot.

    Args:
        track: The track the cars are on.
        car: The cars' parameters (the steering lock scales the steering input).
        config: Which inputs go in, and how far ahead the bends are measured.
        rays: The distance sensors' settings, used when the rays are in.
    """

    def __init__(
        self,
        track: Track,
        car: VehicleParams,
        config: ObservationConfig,
        rays: RaySettings | None = None,
    ) -> None:
        rays = rays or RaySettings()
        self.track = track
        self.car = car
        self.sensor = RaySensor(track, rays) if config.rays else None
        """The distance sensors, or ``None`` when the rays are out."""
        self.readings: RayReadings | None = None
        """What the rays saw for the latest observation, for drawing them."""

        stretch = config.lookahead / config.lookahead_points
        features: dict[str, tuple[Feature, _Compute]] = {
            "rays": (
                Feature(
                    "rays",
                    _numbered("rays", rays.count),
                    0.0,
                    1.0,
                    (
                        ("count", rays.count),
                        ("field_of_view", rays.field_of_view),
                        ("range", rays.max_range),
                    ),
                ),
                self._rays,
            ),
            "speed": (
                Feature("speed", ("speed",), -LIMIT, LIMIT, (("scale", SPEED_SCALE),)),
                self._speed,
            ),
            "heading": (Feature("heading", ("heading.sin", "heading.cos"), -1.0, 1.0), _heading),
            "offset": (Feature("offset", ("offset",), -LIMIT, LIMIT), self._offset),
            "yaw_rate": (
                Feature("yaw_rate", ("yaw_rate",), -LIMIT, LIMIT, (("scale", YAW_RATE_SCALE),)),
                _yaw_rate,
            ),
            "steering": (Feature("steering", ("steering",), -1.0, 1.0), self._steering),
            "previous_action": (
                Feature(
                    "previous_action",
                    ("previous_action.steer", "previous_action.pedal"),
                    -1.0,
                    1.0,
                ),
                _previous_action,
            ),
            "curvature": (
                Feature(
                    "curvature",
                    _numbered("curvature", config.lookahead_points),
                    -LIMIT,
                    LIMIT,
                    (("stretch", stretch), ("scale", CURVATURE_SCALE)),
                ),
                self._curvature,
            ),
        }
        chosen = [features[name] for name in OBSERVATION_INPUTS if getattr(config, name)]
        self.spec = ObservationSpec(tuple(feature for feature, _ in chosen))
        """What each observation holds."""
        self._parts = tuple(compute for _, compute in chosen)
        self._low, self._high = self.spec.low, self.spec.high

        # The road's direction along the lap, without the jumps at +-pi: the bend between two
        # places is the change in direction divided by the distance between them.
        line = track.centerline
        direction = np.unwrap(np.arctan2(line.tangent[:, 1], line.tangent[:, 0]))
        lap_end = direction[-1] + wrap_angle(direction[0] - direction[-1])
        self._direction_at = np.append(line.arc_length, line.length), np.append(direction, lap_end)
        self._lap_turn = float(lap_end - direction[0])  # +-2 pi for a track that doesn't cross
        self._stretches = np.arange(config.lookahead_points + 1) * stretch

    def build(self, snapshot: Snapshot, previous_actions: ArrayLike) -> NDArray[np.float32]:
        """Every car's observation, shape ``(N, spec.size)``.

        Args:
            snapshot: The world now.
            previous_actions: The ``[steer, pedal]`` each car was last given, shape ``(N, 2)``;
                zeros at the start of a run.

        Raises:
            ValueError: If the previous actions have the wrong shape or aren't finite numbers.
        """
        actions = checked_actions(previous_actions, len(snapshot.cars))
        values = np.concatenate([compute(snapshot, actions) for compute in self._parts], axis=1)
        observations: NDArray[np.float32] = np.clip(values, self._low, self._high).astype(
            np.float32
        )
        return observations

    def _rays(self, snapshot: Snapshot, _: FloatArray) -> FloatArray:
        assert self.sensor is not None  # only chosen when the rays are in
        self.readings = self.sensor.sense(snapshot)
        return self.readings.normalized

    def _speed(self, snapshot: Snapshot, _: FloatArray) -> FloatArray:
        return snapshot.cars.vx[:, None] / SPEED_SCALE

    def _offset(self, snapshot: Snapshot, _: FloatArray) -> FloatArray:
        race = snapshot.race
        half_width = self.track.width_at(race.arc_length) / 2
        return (race.offset / half_width)[:, None]

    def _steering(self, snapshot: Snapshot, _: FloatArray) -> FloatArray:
        return snapshot.cars.steer[:, None] / self.car.max_steer

    def _curvature(self, snapshot: Snapshot, _: FloatArray) -> FloatArray:
        """How much the road bends over each stretch ahead of each car: the change in its
        direction over the stretch, per metre, times `CURVATURE_SCALE`."""
        ahead = snapshot.race.arc_length[:, None] + self._stretches  # (N, points + 1)
        arc_length, direction = self._direction_at
        laps = np.floor(ahead / self.track.centerline.length)
        place = ahead - laps * self.track.centerline.length
        turned = np.interp(place, arc_length, direction) + laps * self._lap_turn
        bends: FloatArray = np.diff(turned, axis=1) / np.diff(self._stretches)
        return bends * CURVATURE_SCALE


def _heading(snapshot: Snapshot, _: FloatArray) -> FloatArray:
    # Sine and cosine instead of the angle: no jump when the car turns past backwards.
    error = snapshot.race.heading_error
    return np.column_stack([np.sin(error), np.cos(error)])


def _yaw_rate(snapshot: Snapshot, _: FloatArray) -> FloatArray:
    return snapshot.cars.yaw_rate[:, None] / YAW_RATE_SCALE


def _previous_action(_: Snapshot, actions: FloatArray) -> FloatArray:
    return actions


def _numbered(name: str, count: int) -> tuple[str, ...]:
    return tuple(f"{name}[{index}]" for index in range(count))
