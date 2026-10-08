"""Tests for mlracecar.training.run: training runs, their folders, checkpoints, and resuming.

Skipped where the training libraries aren't installed (``uv sync --extra train`` or
``--extra train-cpu``). The runs are tiny: a few hundred car-steps, a second or two each.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO

from mlracecar.agents.sb3 import SB3Agent
from mlracecar.config.files import load_config
from mlracecar.config.models import RacecarConfig
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.io.track_file import TrackFile, write_track_file
from mlracecar.training.run import TrainingError, TrainingRun

TECHNICAL = Path(__file__).parents[3] / "tracks" / "technical.json"


def tiny(track: Path = TECHNICAL, **changes: Any) -> RacecarConfig:
    """A run of 256 car-steps on 2 cars: four updates, tested and saved every 128 steps."""
    settings: dict[str, Any] = {
        "training": {
            "track": str(track),
            "steps": 256,
            "cars": 2,
            "checkpoint_every": 128,
            "eval_every": 128,
            "eval_runs": 2,
        },
        "ppo": {"steps_per_car": 32, "batch_size": 64, "epochs": 1, "layers": 1, "layer_size": 8},
        "episode": {"time_limit": 3.0},
    }
    for section, values in changes.items():
        settings[section] = settings.get(section, {}) | values
    return RacecarConfig.model_validate(settings)


def results(run: TrainingRun) -> list[dict[str, Any]]:
    """The run's test results without their wall-clock times."""
    return [
        {key: value for key, value in test.items() if key != "seconds"}
        for test in run.evaluations()
    ]


# --------------------------------------------------------------------------- #
# A run and its folder
# --------------------------------------------------------------------------- #


def test_a_run_keeps_everything_in_its_folder(tmp_path: Path) -> None:
    config = tiny()
    lines: list[str] = []

    run = TrainingRun.start(config, tmp_path, name="first lap!")
    outcome = run.train(report=lines.append)

    folder = run.directory
    assert folder.parent == tmp_path
    assert folder.name.endswith("_first-lap")
    assert load_config([folder / "config.yaml"]) == config
    checkpoints = sorted(path.name for path in (folder / "checkpoints").iterdir())
    assert checkpoints == ["best", "last", "step_000000128", "step_000000256"]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    assert meta["steps"] == outcome.steps == 256
    assert meta["seeds"]["cars"] == [0, 1]
    assert len(meta["seeds"]["tests"]) == 2
    assert meta["track"]["path"] == TECHNICAL.as_posix()
    assert meta["git"]["commit"]
    assert meta["hardware"]["device"] == "cpu"
    assert set(meta["versions"]) >= {"python", "torch", "stable-baselines3"}
    [session] = meta["sessions"]
    assert (session["from_step"], session["to_step"]) == (0, 256)
    assert [test["step"] for test in run.evaluations()] == [128, 256]
    assert meta["best"]["score"] == outcome.best_score == max(t["score"] for t in run.evaluations())
    assert lines[0].startswith("step 128/256  score ")
    assert lines[-1].startswith("Finished: 256 steps.")


def test_every_saved_agent_loads_and_fits_the_environment(tmp_path: Path) -> None:
    config = tiny()
    run = TrainingRun.start(config, tmp_path)
    run.train(report=lambda line: None)
    spec = BatchedRacingEnv(TECHNICAL, 1, config).observations.spec

    for folder in (run.directory / "checkpoints").iterdir():
        agent = SB3Agent.load(folder, spec)
        assert agent.card.metadata["run"] == run.meta["name"]
        assert agent.card.metadata["steps"] in {128, 256}


def test_each_test_says_how_its_runs_went(tmp_path: Path) -> None:
    run = TrainingRun.start(tiny(), tmp_path)
    run.train(report=lambda line: None)

    test = run.evaluations()[-1]

    assert test["runs"] == 2
    assert set(test) >= {"score", "distance", "average_speed", "laps", "best_lap", "end_reasons"}
    assert sum(test["end_reasons"].values()) == 2
    assert test["grid"]["seconds"] <= 3.0


def test_the_end_of_a_run_is_always_tested_and_saved(tmp_path: Path) -> None:
    run = TrainingRun.start(tiny(training={"checkpoint_every": 192, "eval_every": 192}), tmp_path)
    assert run.evaluations() == []  # nothing tested yet

    run.train(report=lambda line: None)

    assert [test["step"] for test in run.evaluations()] == [192, 256]
    checkpoints = sorted(path.name for path in (run.directory / "checkpoints").iterdir())
    assert checkpoints == ["best", "last", "step_000000192", "step_000000256"]


# --------------------------------------------------------------------------- #
# Reproducible
# --------------------------------------------------------------------------- #


def test_the_same_settings_and_seed_give_the_same_run(tmp_path: Path) -> None:
    first = TrainingRun.start(tiny(), tmp_path / "a")
    first.train(report=lambda line: None)
    second = TrainingRun.start(tiny(), tmp_path / "b")
    second.train(report=lambda line: None)
    other = TrainingRun.start(tiny(training={"seed": 1}), tmp_path / "c")
    other.train(report=lambda line: None)

    assert results(first) == results(second)
    observations = np.random.default_rng(0).uniform(-1, 1, (4, 31)).astype(np.float32)
    actions = [
        SB3Agent.load(run.directory / "checkpoints" / "last").act(observations)
        for run in (first, second, other)
    ]
    np.testing.assert_array_equal(actions[0], actions[1])  # the same network, to the last bit
    assert not np.array_equal(actions[0], actions[2])  # another seed learns differently


# --------------------------------------------------------------------------- #
# Stopping and carrying on
# --------------------------------------------------------------------------- #


def stop_on_second_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl+C during PPO's second update."""
    real = PPO.train
    calls = {"count": 0}

    def train(self: PPO) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        real(self)

    monkeypatch.setattr(PPO, "train", train)


def test_ctrl_c_saves_the_agent_and_a_resumed_run_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = TrainingRun.start(tiny(), tmp_path)
    lines: list[str] = []
    with monkeypatch.context() as patch:
        stop_on_second_update(patch)
        stopped = run.train(report=lines.append)

    assert stopped.status == "interrupted"
    assert run.meta["status"] == "interrupted"
    resume = f"racecar train --resume {run.directory}"
    assert lines[-1] == f"Stopped at step {stopped.steps:,}. Carry on with: {resume}"
    last = SB3Agent.load(run.directory / "checkpoints" / "last")
    assert last.card.metadata["steps"] == stopped.steps

    resumed = TrainingRun.resume(run.directory)
    finished = resumed.train(report=lambda line: None)

    assert finished.status == "finished"
    assert finished.steps == 256
    assert resumed.meta["status"] == "finished"
    first, second = resumed.meta["sessions"]
    assert (first["to_step"], second["from_step"], second["to_step"]) == (
        stopped.steps,
        stopped.steps,
        256,
    )
    assert [test["step"] for test in resumed.evaluations()][-1] == 256


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"ppo": {"batch_size": 48}}, r"ppo.batch_size \(48\) must divide .* 2 x 32 = 64"),
    ],
)
def test_settings_that_dont_work_together_are_refused(
    changes: dict[str, Any], message: str, tmp_path: Path
) -> None:
    with pytest.raises(TrainingError, match=message):
        TrainingRun.start(tiny(**changes), tmp_path)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())


def test_a_track_with_errors_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    write_track_file(
        TrackFile.from_arrays("Broken", [[0, 0], [10, 0], [5, 8]], [12, 12, 12]), broken
    )

    with pytest.raises(TrainingError, match="the track has errors; `racecar check"):
        TrainingRun.start(tiny(track=broken), tmp_path / "runs")


def test_only_a_stopped_run_can_be_carried_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(TrainingError, match="not a training run"):
        TrainingRun.resume(tmp_path)

    never_trained = TrainingRun.start(tiny(), tmp_path / "a")
    with pytest.raises(TrainingError, match="no saved agent to carry on from"):
        TrainingRun.resume(never_trained.directory)

    finished = TrainingRun.start(tiny(), tmp_path / "b")
    finished.train(report=lambda line: None)
    with pytest.raises(TrainingError, match="this run is finished"):
        TrainingRun.resume(finished.directory)


def test_a_run_wont_carry_on_if_its_track_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track = tmp_path / "track.json"
    shutil.copy(TECHNICAL, track)
    run = TrainingRun.start(tiny(track=track), tmp_path / "runs")
    with monkeypatch.context() as patch:
        stop_on_second_update(patch)
        run.train(report=lambda line: None)
    track.write_text(
        track.read_text(encoding="utf-8").replace('"name"', '"name" ', 1), encoding="utf-8"
    )

    with pytest.raises(TrainingError, match=r"has changed .* since this run started"):
        TrainingRun.resume(run.directory)
