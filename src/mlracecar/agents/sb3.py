"""`SB3Agent`: a trained Stable-Baselines3 model driving cars, behind the `Agent` protocol
(architecture section 4.9, ADR-0006).

Only `agents` and `training` import Stable-Baselines3 and PyTorch, so the rest of MLRacecar runs
without them (``uv sync --extra train`` installs them). Import this module directly; importing
`mlracecar.agents` doesn't load it.

A saved agent is a folder of two files: ``model.zip``, Stable-Baselines3's own file, and
``model_card.json`` (`mlracecar.io.model_card`), which says what the model expects. Loading checks
the card against the environment the agent is meant for, and refuses a model that was trained on
different observations, saying what differs.
"""

import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import JsonValue
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from mlracecar.config.models import RacecarConfig
from mlracecar.env.observations import ObservationSpec, observation_differences
from mlracecar.io.model_card import (
    ActionSpec,
    GitCommit,
    ModelCard,
    read_model_card,
    write_model_card,
)

MODEL_FILE = "model.zip"
"""The model itself, in Stable-Baselines3's format."""

CARD_FILE = "model_card.json"
"""What the model expects, and how it was made."""

ACTIONS = ActionSpec(labels=("steer", "pedal"), low=(-1.0, -1.0), high=(1.0, 1.0))
"""The actions every MLRacecar agent gives."""

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"PPO": PPO, "SAC": SAC}
"""The Stable-Baselines3 algorithms a card can name."""

LIBRARIES = ("mlracecar", "numpy", "gymnasium", "torch", "stable-baselines3")
"""The libraries whose versions a card records."""


class IncompatibleModelError(ValueError):
    """A model that expects different observations or actions than the environment has."""


def make_model_card(
    model: BaseAlgorithm,
    config: RacecarConfig,
    spec: ObservationSpec,
    extra: dict[str, JsonValue] | None = None,
) -> ModelCard:
    """The card for a model trained on observations ``spec`` with settings ``config``.

    Args:
        model: The trained model.
        config: Every setting the environment ran with.
        spec: The observations the model learned from.
        extra: Anything else to keep, such as the number of training steps.

    Raises:
        ValueError: If the model's algorithm isn't one of `ALGORITHMS`.
    """
    algorithm = type(model).__name__
    if algorithm not in ALGORITHMS:
        raise ValueError(f"no model cards for {algorithm} yet: only {', '.join(ALGORITHMS)}")
    return ModelCard(
        algorithm=algorithm,
        created=datetime.now(UTC),
        observation=spec.description(),
        observation_digest=spec.digest,
        action=ACTIONS,
        config=config.model_dump(mode="json"),
        versions=library_versions(),
        git=git_commit(),
        metadata=extra or {},
    )


def check_compatible(card: ModelCard, spec: ObservationSpec, source: str = "model") -> None:
    """Make sure a model fits an environment: the same observations and the same actions.

    Raises:
        IncompatibleModelError: If they differ, listing every difference.
    """
    if card.observation_digest == spec.digest and card.action == ACTIONS:
        return
    problems = observation_differences(card.observation, spec.description())
    if card.action != ACTIONS:
        problems.append(
            f"actions: {list(ACTIONS.labels)} from {ACTIONS.low} to {ACTIONS.high} here, "
            f"{list(card.action.labels)} from {card.action.low} to {card.action.high} in the model"
        )
    if not problems:  # the digest differs, but nothing in the description does: a changed card
        problems.append("the observation digest doesn't match its description")
    raise IncompatibleModelError(
        f"{source}: this model doesn't fit this environment:\n"
        + "\n".join(f"  {problem}" for problem in problems)
    )


class SB3Agent:
    """A trained Stable-Baselines3 model, driving.

    Args:
        model: The model.
        card: What the model expects, and how it was made.
        deterministic: Act on the policy's best guess (its mean action) instead of sampling;
            the same observation then always gives the same action.
    """

    def __init__(self, model: BaseAlgorithm, card: ModelCard, *, deterministic: bool = True):
        self.model = model
        self.card = card
        self.deterministic = deterministic

    def reset(self, seed: int | None = None) -> None:
        """Get ready for a new run. A seed makes sampled actions repeat; it seeds PyTorch's,
        NumPy's, and Python's global random numbers, which Stable-Baselines3 samples from."""
        if seed is not None:
            self.model.set_random_seed(seed)

    def act(self, observations: ArrayLike) -> NDArray[np.float32]:
        """``[steer, pedal]`` for each car, shape ``(n, 2)``, from observations ``(n, obs_dim)``."""
        batch = np.asarray(observations, dtype=np.float32)
        actions, _ = self.model.predict(batch, deterministic=self.deterministic)
        result: NDArray[np.float32] = np.asarray(actions, dtype=np.float32).reshape(len(batch), 2)
        return result

    def save(self, directory: str | Path) -> None:
        """Save the model and its card into ``directory``, making it if needed."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save(folder / MODEL_FILE)
        write_model_card(self.card, folder / CARD_FILE)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        spec: ObservationSpec | None = None,
        *,
        deterministic: bool = True,
        device: str = "cpu",
    ) -> Self:
        """Load an agent saved with `save`.

        Args:
            directory: The folder it was saved in.
            spec: The observations it will be given. If given, the card must match them.
            deterministic: As for the constructor.
            device: Where the network runs: ``"cpu"`` (fastest for small networks) or
                ``"cuda"``.

        Raises:
            ModelCardError: If the card can't be read or isn't valid.
            IncompatibleModelError: If the model expects different observations or actions.
        """
        folder = Path(directory)
        card = read_model_card(folder / CARD_FILE)
        if spec is not None:
            check_compatible(card, spec, source=str(folder))
        model = ALGORITHMS[card.algorithm].load(folder / MODEL_FILE, device=device)
        return cls(model, card, deterministic=deterministic)


def library_versions() -> dict[str, str]:
    """Python's version and each of `LIBRARIES`'s."""
    versions = {"python": platform.python_version()}
    for library in LIBRARIES:
        try:
            versions[library] = metadata.version(library)
        except metadata.PackageNotFoundError:
            versions[library] = "not installed"
    return versions


def git_commit() -> GitCommit | None:
    """The commit the working folder is on, or ``None`` outside a git checkout."""
    try:
        commit = _git("rev-parse", "HEAD")
        changes = _git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return None
    return GitCommit(commit=commit, dirty=bool(changes))


def _git(*arguments: str) -> str:
    done = subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True, timeout=30
    )
    return done.stdout.strip()
