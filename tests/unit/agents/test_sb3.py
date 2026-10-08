"""Tests for mlracecar.agents.sb3: a Stable-Baselines3 model driving, saved with its card.

Skipped where the training libraries aren't installed (``uv sync --extra train`` or
``--extra train-cpu``).
"""

import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import A2C, PPO

import mlracecar.agents.sb3 as sb3
from mlracecar.agents.sb3 import (
    ACTIONS,
    CARD_FILE,
    MODEL_FILE,
    IncompatibleModelError,
    SB3Agent,
    check_compatible,
    make_model_card,
)
from mlracecar.config.models import ObservationConfig, RacecarConfig
from mlracecar.core.sensors import RaySettings
from mlracecar.env.observations import ObservationBuilder, ObservationSpec
from mlracecar.env.racing import RacingEnv
from mlracecar.io.model_card import ActionSpec, ModelCardError

TECHNICAL = Path(__file__).parents[3] / "tracks" / "technical.json"
CONFIG = RacecarConfig()


@pytest.fixture(scope="module")
def env() -> RacingEnv:
    return RacingEnv(TECHNICAL, CONFIG)


@pytest.fixture(scope="module")
def model(env: RacingEnv) -> PPO:
    """A small PPO model, trained just enough that its weights aren't the starting ones."""
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": [16]},
        device="cpu",
        seed=0,
    )
    return model.learn(total_timesteps=128)


@pytest.fixture
def agent(model: PPO, env: RacingEnv) -> SB3Agent:
    return SB3Agent(model, make_model_card(model, CONFIG, env.observations.spec, {"steps": 128}))


def observations(env: RacingEnv, count: int = 5) -> np.ndarray:
    spec = env.observations.spec
    rng = np.random.default_rng(0)
    return rng.uniform(spec.low, spec.high, (count, spec.size)).astype(np.float32)


def spec_with(config: ObservationConfig, rays: RaySettings | None = None) -> ObservationSpec:
    return ObservationBuilder(
        RacingEnv(TECHNICAL).track, CONFIG.vehicle.to_params(), config, rays
    ).spec


# --------------------------------------------------------------------------- #
# Acting
# --------------------------------------------------------------------------- #


def test_it_gives_steer_and_pedal_for_each_car(agent: SB3Agent, env: RacingEnv) -> None:
    actions = agent.act(observations(env))

    assert actions.shape == (5, 2)
    assert actions.dtype == np.float32
    assert ((actions >= -1) & (actions <= 1)).all()


def test_in_deterministic_mode_the_same_observation_gives_the_same_action(
    agent: SB3Agent, env: RacingEnv
) -> None:
    seen = observations(env)

    np.testing.assert_array_equal(agent.act(seen), agent.act(seen))


def test_sampled_actions_repeat_from_a_seed(model: PPO, env: RacingEnv) -> None:
    agent = SB3Agent(
        model, make_model_card(model, CONFIG, env.observations.spec), deterministic=False
    )
    seen = observations(env)

    agent.reset(seed=3)
    first = agent.act(seen)
    agent.reset(seed=3)
    again = agent.act(seen)
    agent.reset()  # no seed: nothing to repeat

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, agent.act(seen))


def test_it_drives_the_environment(agent: SB3Agent, env: RacingEnv) -> None:
    observation, _ = env.reset(seed=0)

    for _ in range(20):
        observation, *_ = env.step(agent.act(observation[None])[0])

    assert env.observation_space.contains(observation)


# --------------------------------------------------------------------------- #
# The model card
# --------------------------------------------------------------------------- #


def test_the_card_records_what_the_model_expects_and_how_it_was_made(
    agent: SB3Agent, env: RacingEnv
) -> None:
    card = agent.card

    assert card.algorithm == "PPO"
    assert card.observation_digest == env.observations.spec.digest
    assert card.observation == env.observations.spec.description()
    assert card.action == ACTIONS
    assert RacecarConfig.model_validate(card.config) == CONFIG
    assert set(card.versions) == {
        "python",
        "mlracecar",
        "numpy",
        "gymnasium",
        "torch",
        "stable-baselines3",
    }
    assert card.git is not None  # the tests run in the project's git checkout
    assert card.metadata == {"steps": 128}


def test_only_known_algorithms_get_cards(env: RacingEnv) -> None:
    model = A2C("MlpPolicy", env, device="cpu")

    with pytest.raises(ValueError, match="no model cards for A2C yet"):
        make_model_card(model, CONFIG, env.observations.spec)


def test_outside_git_the_card_has_no_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def no_git(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)

    assert sb3._git_commit() is None


def test_a_library_that_isnt_installed_is_recorded_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    real = metadata.version

    def version(name: str) -> str:
        if name == "gymnasium":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", version)

    assert sb3._versions()["gymnasium"] == "not installed"


# --------------------------------------------------------------------------- #
# Saving and loading
# --------------------------------------------------------------------------- #


def test_a_loaded_agent_acts_exactly_as_the_saved_one(
    agent: SB3Agent, env: RacingEnv, tmp_path: Path
) -> None:
    agent.save(tmp_path / "agent")

    loaded = SB3Agent.load(tmp_path / "agent", env.observations.spec)

    assert sorted(path.name for path in (tmp_path / "agent").iterdir()) == sorted(
        [CARD_FILE, MODEL_FILE]
    )
    assert loaded.card == agent.card
    seen = observations(env)
    np.testing.assert_array_equal(loaded.act(seen), agent.act(seen))
    np.testing.assert_array_equal(loaded.act(seen), loaded.act(seen))


def test_a_model_for_other_observations_is_refused_with_every_difference(
    agent: SB3Agent, tmp_path: Path
) -> None:
    agent.save(tmp_path)
    other = spec_with(ObservationConfig(curvature=False, lookahead=200.0), RaySettings(count=19))

    with pytest.raises(IncompatibleModelError) as caught:
        SB3Agent.load(tmp_path, other)

    assert str(caught.value).splitlines() == [
        f"{tmp_path}: this model doesn't fit this environment:",
        "  curvature: in the model, but turned off here",
        "  rays: 19 values here, 15 in the model",
        "  rays: count is 19 here, 15 in the model",
    ]


def test_without_a_spec_the_card_isnt_checked(agent: SB3Agent, tmp_path: Path) -> None:
    agent.save(tmp_path)

    assert SB3Agent.load(tmp_path).card == agent.card


def test_other_actions_are_named_too(agent: SB3Agent, env: RacingEnv) -> None:
    card = agent.card.model_copy(
        update={"action": ActionSpec(labels=("throttle",), low=(0.0,), high=(1.0,))}
    )

    with pytest.raises(IncompatibleModelError, match=r"actions: \['steer', 'pedal'\] from"):
        check_compatible(card, env.observations.spec)


def test_a_card_whose_digest_doesnt_match_its_description_is_refused(
    agent: SB3Agent, env: RacingEnv
) -> None:
    card = agent.card.model_copy(update={"observation_digest": "0" * 64})

    with pytest.raises(IncompatibleModelError, match="digest doesn't match its description"):
        check_compatible(card, env.observations.spec)


def test_a_folder_without_a_card_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelCardError, match="can't read the model card"):
        SB3Agent.load(tmp_path)
