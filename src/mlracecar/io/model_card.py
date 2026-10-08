"""Model cards: what every saved model says about itself (ADR-0006).

A trained model is only useful with the observations it learned from. Its card records exactly
that, so a model can't be fed different numbers without anyone noticing, and so a result can be
traced back to how it was made:

- the **observation** it expects: the spec's description and digest
  (`mlracecar.env.observations.ObservationSpec`);
- the **actions** it gives: their names and bounds;
- every **setting** the environment ran with (`RacecarConfig`, as JSON);
- the **versions** of Python and the libraries, and the **git commit** of the code;
- the **algorithm**, when it was made, and free-form **metadata** (training steps, seeds, ...).

Cards are versioned JSON, read and checked like track files: a newer format is refused, and a
broken card gets one error naming every problem.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

CURRENT_VERSION = 1
"""The model card format this version of MLRacecar writes."""


class ModelCardError(ValueError):
    """A model card that can't be read, or isn't a well-formed model card."""


_STRICT = ConfigDict(extra="forbid", frozen=True, strict=True)


class GitCommit(BaseModel):
    """The code a model was made with."""

    model_config = _STRICT

    commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    """The commit's full hash."""
    dirty: bool
    """Whether there were changes not yet committed."""


class ActionSpec(BaseModel):
    """The actions a model gives: one value per label, each within its bounds."""

    model_config = _STRICT

    labels: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]


class ModelCard(BaseModel):
    """Everything a saved model says about itself."""

    model_config = _STRICT

    schema_version: Literal[1] = 1
    algorithm: Annotated[str, Field(min_length=1)]
    """The algorithm that trained it, such as ``"PPO"``."""
    created: datetime
    """When the model was saved."""
    observation: dict[str, JsonValue]
    """The description of the observations it expects (`ObservationSpec.description`)."""
    observation_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    """Their digest (`ObservationSpec.digest`)."""
    action: ActionSpec
    """The actions it gives."""
    config: dict[str, JsonValue]
    """Every setting the environment ran with (`RacecarConfig`, as JSON)."""
    versions: dict[str, str]
    """Python's and the main libraries' versions."""
    git: GitCommit | None
    """The code it was made with, if it was made in a git checkout."""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Anything else worth keeping, such as the number of training steps."""


def read_model_card(path: str | Path) -> ModelCard:
    """Read and check a model card.

    Raises:
        ModelCardError: If the file can't be read or isn't a well-formed model card.
    """
    path = Path(path)
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ModelCardError(f"{path}: can't read the model card ({error.strerror})") from error
    return parse_model_card(content, source=str(path))


def parse_model_card(text: str | bytes, source: str = "model card") -> ModelCard:
    """Check the text of a model card; ``source`` names it in error messages.

    Raises:
        ModelCardError: If the text isn't a well-formed model card.
    """
    try:
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCardError(f"{source}: not valid JSON ({error})") from None
    if not isinstance(data, dict):
        raise ModelCardError(f"{source}: expected a JSON object with the card's fields")
    version = data.get("schema_version")
    if isinstance(version, int) and not isinstance(version, bool) and version > CURRENT_VERSION:
        raise ModelCardError(
            f"{source}: this model was saved with model card version {version}, but this "
            f"version of MLRacecar can only read up to version {CURRENT_VERSION}. "
            "Update MLRacecar to load it."
        )
    try:
        # Checked as JSON: strict mode then reads dates from text and tuples from lists.
        return ModelCard.model_validate_json(json.dumps(data))
    except ValidationError as error:
        problems = [
            f"  {'.'.join(str(part) for part in issue['loc']) or '(top level)'}: {issue['msg']}"
            for issue in error.errors()
        ]
        raise ModelCardError(f"{source}: not a valid model card:\n" + "\n".join(problems)) from None


def format_model_card(card: ModelCard) -> str:
    """The text of a model card: indented JSON, readable in any editor."""
    return json.dumps(card.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def write_model_card(card: ModelCard, path: str | Path) -> None:
    """Save a model card, replacing the file in one step so it's never half written."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(format_model_card(card), encoding="utf-8", newline="\n")
    temporary.replace(path)
