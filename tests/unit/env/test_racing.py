"""Tests for mlracecar.env.racing: the Gymnasium environment, made as users make it."""

import importlib
import math
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import pygame
import pytest
from gymnasium.utils.env_checker import check_env

import mlracecar.env
from drivers import CenterlineDriver
from mlracecar.config.models import EpisodeConfig, RacecarConfig
from mlracecar.core.track.model import Track
from mlracecar.env import ENV_ID
from mlracecar.env.racing import RacingEnv
from mlracecar.render.race import RAY

TRACKS = Path(__file__).parents[3] / "tracks"
TECHNICAL = TRACKS / "technical.json"
ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CIRCLE = Track.build(60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 48)
FULL_THROTTLE = np.array([0.0, 1.0], dtype=np.float32)


def make(**kwargs: Any) -> RacingEnv:
    """The environment as ``gymnasium.make`` gives it, without Gymnasium's wrappers."""
    kwargs.setdefault("track", TECHNICAL)
    env = gymnasium.make(ENV_ID, **kwargs).unwrapped
    assert isinstance(env, RacingEnv)
    return env


def with_episode(**settings: Any) -> RacecarConfig:
    return RacecarConfig(episode=EpisodeConfig.model_validate(settings))


# --------------------------------------------------------------------------- #
# The standard
# --------------------------------------------------------------------------- #


def test_it_is_registered_with_gymnasium() -> None:
    assert ENV_ID in gymnasium.registry


def test_importing_again_registers_it_once_without_a_warning() -> None:
    importlib.reload(mlracecar.env)  # pytest turns any warning into an error

    assert gymnasium.registry[ENV_ID].entry_point == "mlracecar.play.environment:make_racing_env"


def test_it_passes_gymnasiums_own_checks() -> None:
    # Including every render mode, which the checker makes through the registration.
    check_env(make())


def test_actions_are_steer_and_pedal_from_minus_1_to_1() -> None:
    space = make().action_space

    assert isinstance(space, gymnasium.spaces.Box)
    assert space.shape == (2,)
    assert space.dtype == np.float32
    assert space.low.tolist() == [-1.0, -1.0]
    assert space.high.tolist() == [1.0, 1.0]


def test_observations_are_what_the_observation_spec_says() -> None:
    env = make()
    space = env.observation_space

    assert isinstance(space, gymnasium.spaces.Box)
    assert space.shape == (env.observations.spec.size,) == (31,)
    np.testing.assert_array_equal(space.low, env.observations.spec.low)
    np.testing.assert_array_equal(space.high, env.observations.spec.high)


def test_stepping_before_the_first_reset_is_refused() -> None:
    with pytest.raises(gymnasium.error.ResetNeeded):
        make().step(FULL_THROTTLE)


@pytest.mark.parametrize("action", [[math.nan, 0.0], [0.0, 1.0, 0.5]])
def test_impossible_actions_are_refused(action: list[float]) -> None:
    env = make()
    env.reset(seed=0)

    with pytest.raises(ValueError, match="actions"):
        env.step(np.array(action))


# --------------------------------------------------------------------------- #
# Starting a run
# --------------------------------------------------------------------------- #


def test_a_run_starts_on_the_grid_at_rest() -> None:
    env = make()

    observation, info = env.reset(seed=0)

    assert env.world is not None
    np.testing.assert_allclose(env.world.snapshot.cars.position, env.track.start_grid(1).position)
    assert info == {
        "distance": 0.0,
        "speed": 0.0,
        "laps": 0,
        "last_lap": None,
        "best_lap": None,
        "off_track": False,
    }
    assert env.observation_space.contains(observation)


def test_random_starts_follow_the_seed() -> None:
    env = make()

    def start(seed: int) -> list[float]:
        env.reset(seed=seed, options={"start": "random"})
        assert env.world is not None
        return [float(value) for value in env.world.snapshot.cars.position[0]]

    assert start(3) == start(3)
    assert start(3) != start(4)
    assert start(3) != env.track.start_grid(1).position[0].tolist()


def test_the_settings_choose_where_runs_start() -> None:
    env = make(config=with_episode(start="random"))
    env.reset(seed=3)

    assert env.world is not None
    assert (
        env.world.snapshot.cars.position[0].tolist() != env.track.start_grid(1).position[0].tolist()
    )


def test_a_reset_can_change_the_track() -> None:
    env = make()

    observation, _ = env.reset(seed=0, options={"track": TRACKS / "oval.json"})

    assert env.track.length != make().track.length
    assert env.observation_space.contains(observation)
    env.reset(seed=0, options={"track": CIRCLE})  # a track itself works too
    assert env.track is CIRCLE


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


def test_each_step_scores_the_progress_made() -> None:
    env = make()
    env.reset(seed=0)
    distance = 0.0

    for _ in range(40):
        observation, reward, terminated, truncated, info = env.step(FULL_THROTTLE)
        assert reward == pytest.approx(0.1 * (info["distance"] - distance))
        assert info["terms"]["progress"] == pytest.approx(reward)
        distance = info["distance"]

    assert distance > 15  # about 18 m: 2 seconds of full throttle from a standstill
    assert info["speed"] > 10
    assert not terminated
    assert not truncated
    assert env.observation_space.contains(observation)


def test_the_agent_sees_the_action_it_just_took() -> None:
    env = make()
    env.reset(seed=0)

    observation, *_ = env.step(np.array([0.5, 0.75]))

    labels = env.observations.spec.labels
    index = labels.index("previous_action.steer")
    np.testing.assert_allclose(observation[index : index + 2], [0.5, 0.75])


def test_leaving_the_road_ends_the_run_and_reports_its_points() -> None:
    env = make(track=CIRCLE)  # driving straight on, the circle curves away
    env.reset(seed=0)
    rewards, terms = [], []

    for _ in range(300):
        _, reward, terminated, truncated, info = env.step(FULL_THROTTLE)
        rewards.append(reward)
        terms.append(info["terms"])
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert info["end_reason"] == "off_track"
    assert terms[-1]["off_track"] == -10.0
    # The run's points per term are the sums of its steps' points.
    for term, total in info["episode_terms"].items():
        assert total == pytest.approx(sum(step[term] for step in terms))
    assert sum(info["episode_terms"].values()) == pytest.approx(sum(rewards))


def test_a_lap_driven_well_counts_and_its_time_is_reported() -> None:
    env = make(track=CIRCLE)
    env.reset(seed=0)
    assert env.world is not None
    driver = CenterlineDriver(CIRCLE, env.car, 25.0)

    while (info := env.step(driver.act(env.world.snapshot)[0])[4])["laps"] == 0:
        assert "end_reason" not in info

    assert info["best_lap"] == pytest.approx(2 * math.pi * 60 / 25, rel=0.05)
    assert info["last_lap"] == pytest.approx(2 * math.pi * 60 / 25, rel=0.05)


def test_the_time_limit_stops_the_run() -> None:
    env = make(config=with_episode(time_limit=1.0))
    env.reset(seed=0)

    endings = [env.step(FULL_THROTTLE)[2:4] for _ in range(20)]

    assert endings[:-1] == [(False, False)] * 19
    assert endings[-1] == (False, True)


def test_a_car_that_doesnt_move_is_stopped_after_5_seconds() -> None:
    env = make()
    env.reset(seed=0)

    for _ in range(99):
        assert env.step(np.zeros(2))[3] is False
    *_, truncated, info = env.step(np.zeros(2))

    assert truncated
    assert info["end_reason"] == "stuck"


def test_runs_repeat_exactly_from_a_seed() -> None:
    def run(seed: int) -> list[tuple[Any, ...]]:
        env = make()
        env.action_space.seed(seed)
        observation, _ = env.reset(seed=seed, options={"start": "random"})
        steps: list[tuple[Any, ...]] = [tuple(observation)]
        for _ in range(300):
            observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            steps.append((*observation, reward, terminated, truncated))
            if terminated or truncated:
                observation, _ = env.reset()
        return steps

    assert run(7) == run(7)
    assert run(7) != run(8)


def drive_randomly(steps: int) -> int:
    """A random agent drives this many steps, starting over whenever a run ends."""
    env = gymnasium.make(ENV_ID, track=TECHNICAL)
    env.action_space.seed(0)
    env.reset(seed=0)
    runs = 0
    for _ in range(steps):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            runs += 1
            env.reset()
    return runs


def test_a_random_agent_drives_without_errors() -> None:
    assert drive_randomly(2_000) >= 1


@pytest.mark.slow
def test_a_random_agent_drives_10_000_steps_without_errors() -> None:
    assert drive_randomly(10_000) >= 10


# --------------------------------------------------------------------------- #
# Watching
# --------------------------------------------------------------------------- #


def has_color(image: np.ndarray, color: tuple[int, int, int]) -> bool:
    """Whether any pixel is this colour, or nearly (smoothed lines blend at their edges)."""
    return bool((np.abs(image.astype(int) - color).sum(axis=-1) <= 24).any())


def test_rgb_array_gives_pictures_showing_the_rays() -> None:
    env = make(render_mode="rgb_array")
    env.reset(seed=0)

    frame = env.render()

    assert frame is not None
    assert frame.shape == (600, 960, 3)
    assert frame.dtype == np.uint8
    assert has_color(frame, RAY)


def test_human_shows_each_step_in_a_window() -> None:
    env = make(render_mode="human")
    env.reset(seed=0)

    assert env.render() is None
    env.step(FULL_THROTTLE)  # each step shows itself
    surface = pygame.display.get_surface()
    assert surface is not None
    assert has_color(pygame.surfarray.array3d(surface).transpose(1, 0, 2), RAY)

    env.close()
    assert pygame.display.get_surface() is None


def test_rendering_without_a_render_mode_only_warns() -> None:
    env = make()
    env.reset(seed=0)

    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None


def test_nothing_is_drawn_before_the_first_reset() -> None:
    assert make(render_mode="rgb_array").render() is None


def test_a_new_track_gets_a_new_picture() -> None:
    env = make(render_mode="rgb_array")
    env.reset(seed=0)
    first = env.render()

    env.reset(seed=0, options={"track": CIRCLE})

    second = env.render()
    assert first is not None
    assert second is not None
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"render_mode": "rgb_array"}, "drawing needs a viewer"),
        ({"render_mode": "ascii"}, "render_mode must be one of"),
    ],
)
def test_impossible_render_modes_are_refused(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RacingEnv(TECHNICAL, **kwargs)
