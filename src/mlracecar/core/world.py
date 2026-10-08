"""The simulation world: cars on a track, advanced in fixed time steps (architecture section 4.4).

`World` is the one object that changes as the race goes on. Everything it hands out is frozen:
a `Snapshot` is a picture of one moment that the renderer, a replay, or the RL environment can
keep without copying, and nothing they do to it can change the race.

There is no global state. Randomness comes only from the `numpy.random.Generator` the world is
given, so the same seed and the same actions always give exactly the same race.
"""

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import BoolArray
from mlracecar.core.race.events import RaceEvent
from mlracecar.core.race.rules import RaceRules, RaceSettings
from mlracecar.core.race.state import RaceState
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import GridLayout, Pose, Track
from mlracecar.core.vehicle.dynamics import DynamicsModel
from mlracecar.core.vehicle.state import VehicleState

MAX_SPEED = 150.0
"""m/s (540 km/h): faster than any car the race rules expect to keep track of."""


@dataclass(frozen=True)
class Timing:
    """How simulated time moves forward: physics in fixed steps, drivers deciding every few steps.

    Fixed steps make a run come out the same on any computer, however fast it is. Each driver
    decision (an action) is held for `action_repeat` physics steps.
    """

    physics_hz: int
    """Physics steps per simulated second."""
    action_repeat: int
    """Physics steps per driver decision."""

    @property
    def dt(self) -> float:
        """Length of one physics step in seconds."""
        return 1.0 / self.physics_hz

    @property
    def decision_dt(self) -> float:
        """Simulated seconds between driver decisions."""
        return self.action_repeat / self.physics_hz


class StartPosition(StrEnum):
    """Where `World.reset` puts cars. They always start at rest, facing the driving direction."""

    GRID = "grid"
    """Each car on its own spot of the starting grid: car ``i`` on spot ``i``."""
    RANDOM = "random"
    """Anywhere along the lap, anywhere across the road with the whole car on it."""


class World:
    """N cars on a track, simulated together in fixed time steps (ADR-0005).

    The cars are ghosts for now: they drive through each other.

    Args:
        track: The track the cars drive on.
        model: The car physics; all cars are of its kind.
        timing: The physics step and how many steps each driver decision lasts.
        cars: How many cars, at least 1.
        rng: The only source of randomness (for random starts).
        start: Where the cars start.
        settings: The race rules' settings (what happens when a car leaves the road).

    Raises:
        ValueError: If ``cars`` is less than 1.
    """

    def __init__(
        self,
        track: Track,
        model: DynamicsModel,
        timing: Timing,
        cars: int,
        rng: np.random.Generator,
        start: StartPosition = StartPosition.GRID,
        settings: RaceSettings | None = None,
    ) -> None:
        if cars < 1:
            raise ValueError(f"a world needs at least 1 car, got {cars}")
        self.track = track
        self.model = model
        self.timing = timing
        self.rules = RaceRules(track, reach=MAX_SPEED * timing.decision_dt, settings=settings)
        self._rng = rng
        # With more cars than fit on one lap, the grid wraps around; ghost cars don't mind.
        layout = GridLayout(car_length=model.params.length, car_width=model.params.width)
        self._grid = track.start_grid(cars, layout)
        self._tick = 0
        placed = self._placed(np.arange(cars), start)
        self._snapshot = self._take_snapshot(placed, self.rules.start(placed), ())

    def __len__(self) -> int:
        """The number of cars."""
        return len(self._snapshot.cars)

    @property
    def snapshot(self) -> Snapshot:
        """The world as it is now."""
        return self._snapshot

    @property
    def grid(self) -> Pose:
        """The starting grid: car ``i``'s spot is entry ``i``."""
        return self._grid

    def step(self, actions: ArrayLike) -> Snapshot:
        """Drive every car for one driver decision: ``timing.action_repeat`` physics steps.

        Cars whose run is over (``race.out``) stay where they are. Afterwards the race rules
        update every car's race and apply the off-track policy.

        Args:
            actions: ``[steer, pedal]`` for each car, shape ``(N, 2)``; each car keeps its action
                for the whole decision.

        Raises:
            ValueError: If the actions have the wrong shape or aren't finite numbers.
        """
        before = self._snapshot
        cars = before.cars
        for _ in range(self.timing.action_repeat):
            cars = self.model.step(cars, actions, self.timing.dt)
        self._tick += self.timing.action_repeat
        cars = before.cars.where(before.race.out, cars)
        race, events = self.rules.update(before.race, before.cars, cars, before.time, self._time)
        cars, race = self.rules.enforce(cars, race, self.timing.decision_dt)
        return self._take_snapshot(cars, race, events)

    def reset(
        self,
        cars: ArrayLike | None = None,
        *,
        start: StartPosition = StartPosition.GRID,
        pose: Pose | None = None,
    ) -> Snapshot:
        """Put some cars back at the start, at rest. The others carry on as they were.

        The clock keeps running: a reset car starts again, not the world.

        A reset car's race starts afresh too: no laps, and its next lap starts at the line.

        Args:
            cars: Which cars, as a boolean mask of shape ``(N,)``; all of them if ``None``.
            start: Where they go.
            pose: Where they go instead of ``start``: one position and heading per reset car,
                in car order. An RL environment uses it to start each car as if it were alone.

        Raises:
            ValueError: If the mask doesn't have one entry per car, or ``pose`` doesn't have
                one entry per reset car.
        """
        count = len(self)
        mask = np.ones(count, dtype=bool) if cars is None else np.asarray(cars, dtype=bool)
        if mask.shape != (count,):
            raise ValueError(f"expected a mask of shape ({count},), got {mask.shape}")
        chosen = np.flatnonzero(mask)
        if pose is None:
            placed = self._placed(chosen, start)
        elif pose.position.shape != (len(chosen), 2) or pose.heading.shape != (len(chosen),):
            raise ValueError(
                f"expected a pose for each of the {len(chosen)} cars reset, got positions of "
                f"shape {pose.position.shape} and headings of shape {pose.heading.shape}"
            )
        else:
            placed = VehicleState.at_rest(pose.position, pose.heading)
        current = self._snapshot
        merged = _merged(current.cars, placed, mask)
        race = _merged(current.race, self.rules.start(placed), mask)
        return self._take_snapshot(merged, race, ())

    def _placed(self, cars: ArrayLike, start: StartPosition) -> VehicleState:
        """The given cars (by index) at rest at their start."""
        index = np.asarray(cars)
        if start is StartPosition.GRID:
            pose = Pose(self._grid.position[index], self._grid.heading[index])
        else:
            pose = random_poses(self.track, self.model.params.width, self._rng, len(index))
        return VehicleState.at_rest(pose.position, pose.heading)

    @property
    def _time(self) -> float:
        return self._tick / self.timing.physics_hz

    def _take_snapshot(
        self, cars: VehicleState, race: RaceState, events: tuple[RaceEvent, ...]
    ) -> Snapshot:
        for arrays in (cars, race):
            for field in fields(arrays):
                getattr(arrays, field.name).flags.writeable = False
        self._snapshot = Snapshot(self._tick, self._time, cars, race, events)
        return self._snapshot


def _merged[State: (VehicleState, RaceState)](current: State, new: State, mask: BoolArray) -> State:
    """``current`` with the masked cars' entries replaced by ``new``'s, in fresh arrays."""
    columns: dict[str, Any] = {}
    for field in fields(current):
        column = getattr(current, field.name).copy()
        column[mask] = getattr(new, field.name)
        columns[field.name] = column
    return type(current)(**columns)


def random_poses(track: Track, car_width: float, rng: np.random.Generator, count: int) -> Pose:
    """Random starting places: anywhere along the lap, and across the road as far as a car
    ``car_width`` wide stays wholly on it; facing the driving direction.

    Draws first every car's distance along the lap, then every car's place across the road, so
    the same generator state always gives the same places.
    """
    arc_length = rng.uniform(0.0, track.length, count)
    room = np.maximum(track.width_at(arc_length) - car_width, 0.0) / 2
    return track.pose_at(arc_length, rng.uniform(-1.0, 1.0, count) * room)
