"""The smoke training run: `racecar train configs/smoke.yaml`, end to end, twice.

A `slow` test (`uv run pytest -m slow --no-cov`), run by CI on every pull request. About 2,000
car-steps: far too few to learn to drive, but enough for the cars to start moving, so the test
results are real numbers to compare between the two runs.
"""

import json
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mlracecar.cli import app

pytest.importorskip("stable_baselines3")

ROOT = Path(__file__).parents[2]
SMOKE = ROOT / "configs" / "smoke.yaml"
TIME_LIMIT = 60.0
"""Seconds the smoke run may take: the ticket's budget for CI."""


def smoke_run(runs: Path) -> tuple[Path, float]:
    started = time.perf_counter()
    result = CliRunner().invoke(app, ["train", str(SMOKE), "--runs", str(runs)])
    seconds = time.perf_counter() - started
    assert result.exit_code == 0, result.output
    [folder] = runs.iterdir()
    return folder, seconds


def results_of(folder: Path) -> list[dict[str, Any]]:
    lines = (folder / "eval" / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "seconds"} for line in lines]


@pytest.mark.slow
def test_the_smoke_run_is_quick_complete_and_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)  # the settings name the track relative to the project

    first, seconds = smoke_run(tmp_path / "first")
    second, _ = smoke_run(tmp_path / "second")

    assert seconds < TIME_LIMIT
    meta = json.loads((first / "meta.json").read_text(encoding="utf-8"))
    assert (meta["status"], meta["steps"]) == ("finished", 2048)
    assert (first / "checkpoints" / "best" / "model.zip").is_file()
    results = results_of(first)
    assert [test["step"] for test in results] == [1024, 2048]
    assert results[-1]["distance"] > 0  # the cars got somewhere
    assert results == results_of(second)  # the same settings and seed: the same results
