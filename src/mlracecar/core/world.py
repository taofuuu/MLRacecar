"""The simulation world: cars on a track, advanced in fixed time steps (architecture section 4.4).

`World` is the one object that changes as the race goes on. Everything it hands out is frozen:
a `Snapshot` is a picture of one moment that the renderer, a replay, or the RL environment can
keep without copying, and nothing they do to it can change the race.

There is no global state. Randomness comes only from the `numpy.random.Generator` the world is
given, so the same seed and the same actions always give exactly the same race.
"""

from dataclasses import dataclass, fields
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.track.model import GridLayout, Pose, Track
from mlracecar.core.vehicle.dynamics import DynamicsModel
from mlracecar.core.vehicle.state import VehicleState


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


@dataclass(frozen=True)
class Snapshot:
    """The world at one moment. Its arrays are read-only, so it can be kept and shared."""

    tick: int
    """Physics steps since the world began."""
    time: float
    """Simulated seconds since the world began."""
    cars: VehicleState
    """Every car's position and motion."""


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
    ) -> None:
        if cars < 1:
            raise ValueError(f"a world needs at least 1 car, got {cars}")
        self.track = track
        self.model = model
        self.timing = timing
        self._rng = rng
        # With more cars than fit on one lap, the grid wraps around; ghost cars don't mind.
        layout = GridLayout(car_length=model.params.length, car_width=model.params.width)
        self._grid = track.start_grid(cars, layout)
        self._tick = 0
        self._snapshot = self._take_snapshot(self._placed(np.arange(cars), start))

    def __len__(self) -> int:
        """The number of cars."""
        return len(self._snapshot.cars)

    @property
    def snapshot(self) -> Snapshot:
        """The world as it is now."""
        return self._snapshot

    def step(self, actions: ArrayLike) -> Snapshot:
        """Drive every car for one driver decision: ``timing.action_repeat`` physics steps.

        Args:
            actions: ``[steer, pedal]`` for each car, shape ``(N, 2)``; each car keeps its action
                for the whole decision.

        Raises:
            ValueError: If the actions have the wrong shape or aren't finite numbers.
        """
        cars = self._snapshot.cars
        for _ in range(self.timing.action_repeat):
            cars = self.model.step(cars, actions, self.timing.dt)
        self._tick += self.timing.action_repeat
        return self._take_snapshot(cars)

    def reset(
        self, cars: ArrayLike | None = None, *, start: StartPosition = StartPosition.GRID
    ) -> Snapshot:
        """Put some cars back at the start, at rest. The others carry on as they were.

        The clock keeps running: a reset car starts again, not the world.

        Args:
            cars: Which cars, as a boolean mask of shape ``(N,)``; all of them if ``None``.
            start: Where they go.

        Raises:
            ValueError: If the mask doesn't have one entry per car.
        """
        count = len(self)
        mask = np.ones(count, dtype=bool) if cars is None else np.asarray(cars, dtype=bool)
        if mask.shape != (count,):
            raise ValueError(f"expected a mask of shape ({count},), got {mask.shape}")
        placed = self._placed(np.flatnonzero(mask), start)
        current = self._snapshot.cars
        columns = {}
        for field in fields(VehicleState):
            column = getattr(current, field.name).copy()
            column[mask] = getattr(placed, field.name)
            columns[field.name] = column
        return self._take_snapshot(VehicleState(**columns))

    def _placed(self, cars: ArrayLike, start: StartPosition) -> VehicleState:
        """The given cars (by index) at rest at their start."""
        index = np.asarray(cars)
        if start is StartPosition.GRID:
            pose = Pose(self._grid.position[index], self._grid.heading[index])
        else:
            pose = self._random_poses(len(index))
        return VehicleState.at_rest(pose.position, pose.heading)

    def _random_poses(self, count: int) -> Pose:
        """Anywhere along the lap, and across the road as far as the whole car stays on it."""
        arc_length = self._rng.uniform(0.0, self.track.length, count)
        room = np.maximum(self.track.width_at(arc_length) - self.model.params.width, 0.0) / 2
        return self.track.pose_at(arc_length, self._rng.uniform(-1.0, 1.0, count) * room)

    def _take_snapshot(self, cars: VehicleState) -> Snapshot:
        for field in fields(cars):
            getattr(cars, field.name).flags.writeable = False
        self._snapshot = Snapshot(self._tick, self._tick / self.timing.physics_hz, cars)
        return self._snapshot
