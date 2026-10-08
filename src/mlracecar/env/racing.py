"""`RacingEnv`: one car on a track, as a standard Gymnasium environment (architecture 4.8).

It ties the pieces together. Each step it drives the `World` one decision forward with the
agent's ``[steer, pedal]``, scores the step (`mlracecar.env.rewards`), checks whether the run
ended (`mlracecar.env.episodes`), and builds what the agent sees next
(`mlracecar.env.observations`). Everything it needs comes from one `RacecarConfig`.

Make one with ``gymnasium.make("MLRacecar-v0", track="tracks/technical.json")``; add
``render_mode="human"`` to watch it in a window, or ``"rgb_array"`` for pictures (videos). The
environment itself never draws: training runs without pygame. Drawing is handed to a viewer
that `mlracecar.play.environment` supplies when a render mode is asked for.
"""

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray

from mlracecar.config.models import RacecarConfig
from mlracecar.core.geometry import FloatArray
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.dynamics import checked_actions
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.core.world import StartPosition, World
from mlracecar.env.episodes import EpisodeRules
from mlracecar.env.observations import ObservationBuilder
from mlracecar.env.rewards import RewardFunction, Rewards, RewardTally
from mlracecar.io.track_file import read_track_file

RENDER_MODES = ("human", "rgb_array")


class Viewer(Protocol):
    """Shows a race: as pictures (``rgb_array``), or in a window (``human``)."""

    def render(self, snapshot: Snapshot, rays: FloatArray | None) -> NDArray[np.uint8] | None:
        """Show the race now: an RGB picture ``(height, width, 3)``, or ``None`` for a window."""
        ...

    def close(self) -> None:
        """Let go of the window, if there is one."""
        ...


type ViewerFactory = Callable[[Track, VehicleParams, str, float], Viewer]
"""Makes a viewer for a track, a car, a render mode, and frames per second."""

type TrackSource = Track | str | os.PathLike[str]
"""A track, or the path of a track file."""


class RacingEnv(gymnasium.Env[NDArray[np.float32], NDArray[np.float32]]):
    """One car on a track: drive it round as far and as fast as it can, staying on the road.

    Args:
        track: The track, or the path of a track file.
        config: Every setting: the car, the timing, the race rules, the sensors, what the agent
            sees, the reward, and when runs end. The defaults if ``None``.
        render_mode: ``"human"``, ``"rgb_array"``, or ``None`` not to draw.
        viewer: Makes the viewer that draws; needed for a render mode.

    Raises:
        ValueError: If the render mode isn't one of `RENDER_MODES`, or there's a render mode
            but no viewer.
    """

    metadata: dict[str, Any] = {"render_modes": list(RENDER_MODES), "render_fps": 20}  # noqa: RUF012

    def __init__(
        self,
        track: TrackSource,
        config: RacecarConfig | None = None,
        *,
        render_mode: str | None = None,
        viewer: ViewerFactory | None = None,
    ) -> None:
        if render_mode is not None and render_mode not in RENDER_MODES:
            raise ValueError(
                f"render_mode must be one of {RENDER_MODES} or None, got {render_mode!r}"
            )
        if render_mode is not None and viewer is None:
            raise ValueError(
                "drawing needs a viewer: make the environment with "
                "gymnasium.make('MLRacecar-v0', track=..., render_mode=...)"
            )
        self.config = config or RacecarConfig()
        self.car = self.config.vehicle.to_params()
        self.timing = self.config.simulation.to_timing()
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 1 / self.timing.decision_dt}
        self._make_viewer = viewer
        self._viewer: Viewer | None = None
        self.rewards = RewardFunction(self.config.reward, self.timing.decision_dt)
        self.episodes = EpisodeRules(self.config.episode, self.timing.decision_dt, cars=1)
        self.tally = RewardTally(1)
        self._use(track)
        spec = self.observations.spec
        self.observation_space = spaces.Box(spec.low, spec.high, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.world: World | None = None
        """The world of the current run; ``None`` until `reset`."""
        self._previous_action = np.zeros((1, 2))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start a new run.

        Args:
            seed: Seeds the environment's randomness (where random starts are), for runs that
                can be repeated exactly.
            options: ``track`` (a track or the path of one) to change track, and ``start``
                (``"grid"`` or ``"random"``) to start somewhere other than `episode.start`.
        """
        super().reset(seed=seed)
        options = options or {}
        if "track" in options:
            self._use(options["track"])
        start = StartPosition(options.get("start", self.config.episode.start))
        self.world = World(
            self.track,
            KinematicBicycle(self.car),
            self.timing,
            1,
            self.np_random,
            start=start,
            settings=self.config.race.to_settings(),
        )
        self._previous_action = np.zeros((1, 2))
        self.episodes.start()
        self.tally.clear()
        snapshot = self.world.snapshot
        observation = self.observations.build(snapshot, self._previous_action)[0]
        if self.render_mode == "human":
            self.render()
        return observation, self._info(snapshot)

    def step(
        self, action: ArrayLike
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Drive one decision (`Timing.decision_dt`) with ``action = [steer, pedal]``.

        Raises:
            gymnasium.error.ResetNeeded: Before the first `reset`.
            ValueError: If the action isn't two finite numbers.
        """
        if self.world is None:
            raise gymnasium.error.ResetNeeded("call reset() before step()")
        action_now = checked_actions(np.asarray(action, dtype=np.float64).reshape(1, -1), 1)
        before = self.world.snapshot
        after = self.world.step(action_now)
        rewards = self.rewards(before, after, action_now, self._previous_action)
        endings = self.episodes.check(after)
        self.tally.add(rewards)
        self._previous_action = action_now
        observation = self.observations.build(after, action_now)[0]
        info = self._info(after, rewards)
        reason = endings.reasons[0]
        if reason is not None:
            info["end_reason"] = reason.value
            info["episode_terms"] = self.tally.of(0)
        if self.render_mode == "human":
            self.render()
        terminated, truncated = bool(endings.terminated[0]), bool(endings.truncated[0])
        return observation, float(rewards.total[0]), terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        """Draw the race now: a picture for ``rgb_array``, the window for ``human``."""
        if self.render_mode is None or self._make_viewer is None:
            gymnasium.logger.warn("render() needs a render_mode, given when the env is made")
            return None
        if self.world is None:
            return None
        if self._viewer is None:
            self._viewer = self._make_viewer(
                self.track, self.car, self.render_mode, self.metadata["render_fps"]
            )
        readings = self.observations.readings
        return self._viewer.render(self.world.snapshot, None if readings is None else readings.end)

    def close(self) -> None:
        """Close the window, if one is open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _use(self, track: TrackSource) -> None:
        """Drive on this track from the next run on."""
        self.track = track if isinstance(track, Track) else read_track_file(Path(track)).to_track()
        self.observations = ObservationBuilder(
            self.track, self.car, self.config.observation, self.config.sensors.to_settings()
        )
        self.close()  # a viewer draws one track

    def _info(self, snapshot: Snapshot, rewards: Rewards | None = None) -> dict[str, Any]:
        """What's worth knowing besides the observation, for logs and evaluation."""
        race = snapshot.race
        info: dict[str, Any] = {
            "distance": float(race.distance[0]),
            "speed": float(snapshot.cars.speed[0]),
            "laps": int(race.laps[0]),
            "best_lap": float(race.best_lap[0]),
            "off_track": bool(race.off_track[0]),
        }
        if rewards is not None:
            info["terms"] = {term: float(points[0]) for term, points in rewards.terms.items()}
        if not math.isfinite(info["best_lap"]):
            info["best_lap"] = None
        return info
