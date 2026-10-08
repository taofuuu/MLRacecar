"""`BatchedRacingEnv`: many cars at once, as a Gymnasium vector environment (ADR-0005).

N copies of `RacingEnv` would be N worlds of one car each, stepped one after another. Here all N
cars share one `World` and are moved, scored, and observed together with NumPy, so N runs cost
little more than one. The cars are ghosts that drive through each other, and each starts as if
it were alone (on pole position, or at a random place drawn from its own seed). So every car's
run is exactly what a `RacingEnv` with the same seed would give; a test checks that against
Gymnasium's `SyncVectorEnv` of N single environments.

When a car's run ends it starts again by itself while the others carry on, in one of
Gymnasium's two ways (``autoreset_mode``):

- ``NEXT_STEP`` (the default): the step that ends a run returns its last observation; the car's
  next step restarts it instead of driving, with a reward of 0.
- ``SAME_STEP``: the step that ends a run restarts the car straight away and returns the new
  run's first observation; the last one is in ``info["final_obs"]`` and the run's last info in
  ``info["final_info"]``. Stable-Baselines3 works this way.

``info`` follows Gymnasium's vector convention: each key holds one entry per car, and
``info["_key"]`` says which cars have it.
"""

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.vector import AutoresetMode
from gymnasium.vector.utils import batch_space
from numpy.typing import ArrayLike, NDArray

from mlracecar.config.models import REWARD_TERMS, RacecarConfig
from mlracecar.core.geometry import BoolArray, FloatArray
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Pose
from mlracecar.core.vehicle.dynamics import checked_actions
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.world import StartPosition, World, random_poses
from mlracecar.env.episodes import EndReason, EpisodeRules
from mlracecar.env.observations import ObservationBuilder
from mlracecar.env.racing import TrackSource, load_track
from mlracecar.env.rewards import RewardFunction, Rewards, RewardTally

AUTORESET_MODES = (AutoresetMode.NEXT_STEP, AutoresetMode.SAME_STEP)
"""The ways a car whose run ended can start again."""

type _Observations = NDArray[np.float32]


class BatchedRacingEnv(gymnasium.vector.VectorEnv[_Observations, NDArray[np.float32], Any]):
    """``num_envs`` cars on one track, each in its own run, simulated together.

    Make one with ``gymnasium.make_vec("MLRacecar-v0", num_envs=64,
    vectorization_mode="vector_entry_point", track=...)``, or directly.

    Args:
        track: The track, or the path of a track file.
        num_envs: How many cars (environments).
        config: Every setting, as for `RacingEnv`. The defaults if ``None``.
        autoreset_mode: When a car whose run ended starts again: one of `AUTORESET_MODES`.

    Raises:
        ValueError: If there are no cars, or the autoreset mode isn't one of `AUTORESET_MODES`.
    """

    def __init__(
        self,
        track: TrackSource,
        num_envs: int,
        config: RacecarConfig | None = None,
        *,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"need at least 1 car, got {num_envs}")
        mode = AutoresetMode(autoreset_mode)
        if mode not in AUTORESET_MODES:
            raise ValueError(f"autoreset_mode must be NEXT_STEP or SAME_STEP, got {mode.name}")
        self.num_envs = num_envs
        self.autoreset_mode = mode
        self.config = config or RacecarConfig()
        self.car = self.config.vehicle.to_params()
        self.timing = self.config.simulation.to_timing()
        self.metadata = {"autoreset_mode": mode, "render_modes": []}
        self.render_mode = None
        self.rewards = RewardFunction(self.config.reward, self.timing.decision_dt)
        self.episodes = EpisodeRules(self.config.episode, self.timing.decision_dt, num_envs)
        self.tally = RewardTally(num_envs)
        self._use(track)
        spec = self.observations.spec
        self.single_observation_space = spaces.Box(spec.low, spec.high, dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.world: World | None = None
        """All the cars' world; ``None`` until `reset`."""
        self.stepped: Snapshot | None = None
        """The world as the latest `reset` or `step` left every car's run, before any car
        started again, with what happened in the step (its events). `world`'s snapshot is the
        same but for the cars that started again, and their restart drops the step's events."""
        self._generators: list[np.random.Generator | None] = [None] * num_envs
        self._previous = np.zeros((num_envs, 2))
        self._restarting = np.zeros(num_envs, dtype=bool)  # NEXT_STEP: runs that ended last step

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[_Observations, dict[str, Any]]:
        """Start every car's run again.

        Args:
            seed: One seed for all (car ``i`` gets ``seed + i``, as in Gymnasium's vector
                environments), a seed per car, or ``None`` to carry on with each car's own
                random generator.
            options: ``track`` and ``start``, as for `RacingEnv.reset`, for every car.

        Raises:
            ValueError: If there's a list of seeds of the wrong length.
        """
        self._seed(seed)
        options = options or {}
        if "track" in options:
            self._use(options["track"])
        start = StartPosition(options.get("start", self.config.episode.start))
        self.world = World(
            self.track,
            KinematicBicycle(self.car),
            self.timing,
            self.num_envs,
            np.random.default_rng(0),  # unused: every car is placed explicitly
            settings=self.config.race.to_settings(),
        )
        everyone = np.ones(self.num_envs, dtype=bool)
        snapshot = self.world.reset(everyone, pose=self._start_poses(everyone, start))
        self.stepped = snapshot
        self._previous[:] = 0.0
        self._restarting[:] = False
        self.episodes.start()
        self.tally.clear()
        return self.observations.build(snapshot, self._previous), _state_info(snapshot, everyone)

    def step(
        self, actions: ArrayLike
    ) -> tuple[_Observations, FloatArray, BoolArray, BoolArray, dict[str, Any]]:
        """Drive every car one decision with ``actions`` of shape ``(num_envs, 2)``.

        Raises:
            gymnasium.error.ResetNeeded: Before the first `reset`.
            ValueError: If the actions have the wrong shape or aren't finite numbers.
        """
        if self.world is None:
            raise gymnasium.error.ResetNeeded("call reset() before step()")
        actions_now = checked_actions(actions, self.num_envs)
        before = self.world.snapshot
        after = self.world.step(actions_now)
        self.stepped = after
        rewards = self.rewards(before, after, actions_now, self._previous)
        endings = self.episodes.check(after)

        # NEXT_STEP: cars whose run ended last step start again instead of driving this step,
        # which is worth nothing to them.
        driving = ~self._restarting
        terms = {term: np.where(driving, points, 0.0) for term, points in rewards.terms.items()}
        total: FloatArray = np.where(driving, rewards.total, 0.0)
        terminated = endings.terminated & driving
        truncated = endings.truncated & driving
        ended = terminated | truncated
        self.tally.add(Rewards(total, terms))
        self._previous = actions_now.copy()
        observations = self.observations.build(after, self._previous)

        info: dict[str, Any] = {}
        end = _end_info(ended, endings.reasons, self.tally) if ended.any() else {}
        if self.autoreset_mode is AutoresetMode.SAME_STEP:
            starting = ended
            if ended.any():
                final_obs = np.full(self.num_envs, None, dtype=object)
                for car in np.flatnonzero(ended):
                    final_obs[car] = observations[car].copy()
                final_info = _state_info(after, ended) | _masked("terms", terms, ended) | end
                info |= {"final_obs": final_obs, "_final_obs": ended.copy()}
                info |= {"final_info": final_info, "_final_info": ended.copy()}
        else:
            starting = self._restarting
            info |= end
            self._restarting = ended

        snapshot = after
        if starting.any():
            snapshot = self.world.reset(starting, pose=self._start_poses(starting))
            self.episodes.start(starting)
            self.tally.clear(starting)
            self._previous[starting] = 0.0
            observations[starting] = self.observations.build(
                snapshot.select(starting), self._previous[starting]
            )
        everyone = np.ones(self.num_envs, dtype=bool)
        info |= _state_info(snapshot, everyone) | _masked("terms", terms, ~starting)
        return observations, total, terminated, truncated, info

    def _seed(self, seed: int | list[int | None] | None) -> None:
        """Seed each car's random generator, as Gymnasium's vector environments do."""
        if seed is None:
            seeds: list[int | None] = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + car for car in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(f"expected {self.num_envs} seeds, one per car, got {len(seeds)}")
        for car, car_seed in enumerate(seeds):
            if car_seed is not None or self._generators[car] is None:
                self._generators[car] = seeding.np_random(car_seed)[0]

    def _start_poses(self, cars: BoolArray, start: StartPosition | None = None) -> Pose:
        """Where these cars start, each as if it were alone on the track."""
        assert self.world is not None
        chosen = np.flatnonzero(cars)
        if (start or StartPosition(self.config.episode.start)) is StartPosition.GRID:
            pole = self.world.grid
            return Pose(
                np.repeat(pole.position[:1], len(chosen), axis=0),
                np.repeat(pole.heading[:1], len(chosen)),
            )
        places = []
        for car in chosen:
            generator = self._generators[car]
            assert generator is not None  # every car's is made at the first reset
            places.append(random_poses(self.track, self.car.width, generator, 1))
        return Pose(
            np.concatenate([place.position for place in places]),
            np.concatenate([place.heading for place in places]),
        )

    def _use(self, track: TrackSource) -> None:
        """Drive on this track from the next reset on."""
        self.track = load_track(track)
        self.observations = ObservationBuilder(
            self.track, self.car, self.config.observation, self.config.sensors.to_settings()
        )


def _state_info(snapshot: Snapshot, cars: BoolArray) -> dict[str, Any]:
    """The cars' distance, speed, laps, latest and best lap (NaN before the first), and
    off-track flag."""
    race = snapshot.race
    values = {
        "distance": race.distance.copy(),
        "speed": snapshot.cars.speed,
        "laps": race.laps.copy(),
        "last_lap": race.last_lap.copy(),
        "best_lap": race.best_lap.copy(),
        "off_track": race.off_track.copy(),
    }
    info: dict[str, Any] = {}
    for key, value in values.items():
        info[key], info[f"_{key}"] = value, cars.copy()
    return info


def _masked(key: str, values: dict[str, FloatArray], cars: BoolArray) -> dict[str, Any]:
    """A dictionary of per-car arrays under ``key``, for these cars."""
    inner: dict[str, Any] = {}
    for name, value in values.items():
        inner[name], inner[f"_{name}"] = value.copy(), cars.copy()
    return {key: inner, f"_{key}": cars.copy()}


def _end_info(
    ended: BoolArray, reasons: tuple[EndReason | None, ...], tally: RewardTally
) -> dict[str, Any]:
    """For the cars whose run just ended: why, and the run's points per term."""
    end_reason = np.full(len(ended), None, dtype=object)
    for car in np.flatnonzero(ended):
        reason = reasons[car]
        end_reason[car] = None if reason is None else reason.value
    sums = {term: tally.sums[term] for term in REWARD_TERMS}
    return {"end_reason": end_reason, "_end_reason": ended.copy()} | _masked(
        "episode_terms", sums, ended
    )
