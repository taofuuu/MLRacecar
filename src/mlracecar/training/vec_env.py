"""`SB3VecEnv`: the many-cars environment as Stable-Baselines3 expects it (ADR-0005, ADR-0006).

Stable-Baselines3 trains on its own vector environment interface, `VecEnv`. This adapter puts
a `BatchedRacingEnv` behind it, so one world of many cars feeds training directly, as fast as it
runs on its own, instead of through Stable-Baselines3's `DummyVecEnv` of separate environments.

Stable-Baselines3 restarts a finished run in the same step, so the environment must use
Gymnasium's ``SAME_STEP`` autoreset. Each car's ``info`` is turned into the dictionary
Stable-Baselines3 expects: the run's last observation in ``terminal_observation`` when it ends,
and ``TimeLimit.truncated`` when it was stopped (the time limit, or stuck) rather than over, so
the learner still counts on what would have followed. A test checks every transition against
`DummyVecEnv` of single `RacingEnv`s.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium.vector import AutoresetMode
from numpy.typing import NDArray
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn

from mlracecar.env.batched import BatchedRacingEnv


class SB3VecEnv(VecEnv):
    """A `BatchedRacingEnv` as a Stable-Baselines3 `VecEnv`.

    Args:
        env: The environment. Its autoreset mode must be ``SAME_STEP``.

    Raises:
        ValueError: If the environment restarts finished runs on the next step instead.
    """

    def __init__(self, env: BatchedRacingEnv) -> None:
        if env.autoreset_mode is not AutoresetMode.SAME_STEP:
            raise ValueError(
                "Stable-Baselines3 restarts finished runs in the same step: make the environment "
                "with autoreset_mode=AutoresetMode.SAME_STEP"
            )
        self.env = env  # first: the base class asks it for its render mode
        super().__init__(env.num_envs, env.single_observation_space, env.single_action_space)
        self._actions = np.zeros((env.num_envs, 2), dtype=np.float32)

    def reset(self) -> NDArray[np.float32]:
        """Start every car's run again, with the seeds and options set since the last reset."""
        seeds = None if all(seed is None for seed in self._seeds) else list(self._seeds)
        options = [options for options in self._options if options]
        if any(other != options[0] for other in options[1:]):
            raise ValueError("every car shares one world: give them all the same reset options")
        observations, info = self.env.reset(seed=seeds, options=options[0] if options else None)
        self.reset_infos = [_car(info, car) for car in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        """Remember the actions for `step_wait`."""
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        """Drive every car one decision, restarting the runs that end."""
        observations, rewards, terminated, truncated, info = self.env.step(self._actions)
        dones = terminated | truncated
        infos: list[dict[str, Any]] = []
        for car in range(self.num_envs):
            if dones[car]:
                # The run that ended: its last step's info. The new run's is the reset info.
                car_info = _car(info["final_info"], car)
                car_info["terminal_observation"] = info["final_obs"][car]
                self.reset_infos[car] = _car(info, car)
            else:
                car_info = _car(info, car)
            car_info["TimeLimit.truncated"] = bool(truncated[car] and not terminated[car])
            infos.append(car_info)
        return observations, rewards.astype(np.float32), dones, infos

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """The environment's attribute, once for each car asked for (they share it)."""
        return [getattr(self.env, attr_name) for _ in self._cars(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set an attribute of the environment, which every car shares."""
        setattr(self.env, attr_name, value)

    def env_method(
        self, method_name: str, *args: Any, indices: VecEnvIndices = None, **kwargs: Any
    ) -> list[Any]:
        """Call a method of the environment once; its result, once for each car asked for."""
        result = getattr(self.env, method_name)(*args, **kwargs)
        return [result for _ in self._cars(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
        """Never: the cars' environment isn't wrapped in Gymnasium wrappers."""
        return [False for _ in self._cars(indices)]

    def _cars(self, indices: VecEnvIndices) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return list(indices)


def _car(info: dict[str, Any], car: int) -> dict[str, Any]:
    """One car's entries of a vector ``info`` (the ones its ``_key`` masks say it has), as plain
    Python values."""
    entries: dict[str, Any] = {}
    for key, value in info.items():
        if key.startswith("_") or not info.get(f"_{key}", np.zeros(car + 1, dtype=bool))[car]:
            continue
        entries[key] = _car(value, car) if isinstance(value, dict) else _plain(value[car])
    return entries


def _plain(value: Any) -> Any:
    """A NumPy scalar as the Python value it holds; anything else as it is."""
    return value.item() if isinstance(value, np.generic) else value
