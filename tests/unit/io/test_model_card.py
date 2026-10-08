"""Tests for mlracecar.io.model_card: reading, checking, and writing model cards."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mlracecar.io.model_card import (
    ActionSpec,
    GitCommit,
    ModelCard,
    ModelCardError,
    format_model_card,
    parse_model_card,
    read_model_card,
    write_model_card,
)

CARD = ModelCard(
    algorithm="PPO",
    created=datetime(2026, 10, 8, 12, 0, tzinfo=UTC),
    observation={"version": 1, "features": [{"name": "speed", "labels": ["speed"]}]},
    observation_digest="ab" * 32,
    action=ActionSpec(labels=("steer", "pedal"), low=(-1.0, -1.0), high=(1.0, 1.0)),
    config={"vehicle": {"mass": 1300.0}},
    versions={"python": "3.12.3", "torch": "2.14.1+cu130"},
    git=GitCommit(commit="0123456789abcdef0123456789abcdef01234567", dirty=False),
    metadata={"steps": 4096},
)


def test_a_written_card_reads_back_the_same(tmp_path: Path) -> None:
    path = tmp_path / "model_card.json"

    write_model_card(CARD, path)

    assert read_model_card(path) == CARD
    assert list(tmp_path.iterdir()) == [path]  # the temporary file was renamed into place


def test_cards_are_readable_json(tmp_path: Path) -> None:
    text = format_model_card(CARD)

    assert text.startswith('{\n  "schema_version": 1,\n  "algorithm": "PPO",')
    assert json.loads(text)["git"] == {
        "commit": "0123456789abcdef0123456789abcdef01234567",
        "dirty": False,
    }


def test_a_card_made_outside_git_says_so() -> None:
    card = CARD.model_copy(update={"git": None})

    assert parse_model_card(format_model_card(card)).git is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{nope", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"schema_version": 2}', "card version 2, but this version of MLRacecar can only read"),
    ],
)  # fmt: skip
def test_broken_cards_are_refused_with_the_reason(text: str, message: str) -> None:
    with pytest.raises(ModelCardError, match=message):
        parse_model_card(text, source="card.json")


def test_every_problem_in_a_card_is_listed() -> None:
    data = json.loads(format_model_card(CARD))
    data["observation_digest"] = "not a digest"
    data["git"]["commit"] = "123"
    del data["algorithm"]

    with pytest.raises(ModelCardError) as caught:
        parse_model_card(json.dumps(data), source="card.json")

    message = str(caught.value)
    assert message.startswith("card.json: not a valid model card:")
    for field in ("algorithm", "observation_digest", "git.commit"):
        assert f"  {field}:" in message


def test_an_unreadable_card_says_so(tmp_path: Path) -> None:
    with pytest.raises(ModelCardError, match="can't read the model card"):
        read_model_card(tmp_path / "missing.json")
