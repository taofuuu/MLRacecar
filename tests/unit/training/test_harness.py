"""Tests for mlracecar.training.harness: scoring agents on tracks, and the reports.

The agents are trained for a few hundred car-steps, once for the whole module: far too little to
drive well, but they're real saved agents with real model cards.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("stable_baselines3")

from mlracecar.agents.sb3 import IncompatibleModelError
from mlracecar.config.files import ConfigError
from mlracecar.config.models import RacecarConfig
from mlracecar.io.model_card import ModelCardError
from mlracecar.io.track_file import TrackFile, write_track_file
from mlracecar.training.harness import (
    EvaluationError,
    evaluate,
    find_tracks,
    format_report,
    load_entrants,
    markdown,
    write_report,
)
from mlracecar.training.run import TrainingRun

ROOT = Path(__file__).parents[3]
TECHNICAL = ROOT / "tracks" / "technical.json"
OVAL = ROOT / "tracks" / "oval.json"


def tiny(seed: int, time_limit: float) -> RacecarConfig:
    return RacecarConfig.model_validate(
        {
            "training": {
                "track": str(TECHNICAL),
                "steps": 128,
                "cars": 2,
                "seed": seed,
                "checkpoint_every": 128,
                "eval_every": 128,
                "eval_runs": 2,
            },
            "ppo": {
                "steps_per_car": 32,
                "batch_size": 64,
                "epochs": 1,
                "layers": 1,
                "layer_size": 8,
            },
            "episode": {"time_limit": time_limit},
        }
    )


@pytest.fixture(scope="module")
def agents(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two saved agents: different seeds, and runs of 3 and 4 seconds."""
    folders = []
    for seed, time_limit in ((0, 3.0), (1, 4.0)):
        run = TrainingRun.start(tiny(seed, time_limit), tmp_path_factory.mktemp("runs"))
        run.train(report=lambda line: None)
        folders.append(run.directory / "checkpoints" / "best")
    return folders[0], folders[1]


def report_of(paths: list[Path], **options: Any) -> dict[str, Any]:
    overrides = options.pop("overrides", [])
    tracks = options.pop("tracks", [TECHNICAL])
    return evaluate(
        load_entrants(paths, overrides), tracks, episodes=4, overrides=overrides, **options
    )


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def test_the_same_agents_and_seed_give_the_same_report(agents: tuple[Path, Path]) -> None:
    first = report_of([agents[0]], tracks=[TECHNICAL, OVAL])
    second = report_of([agents[0]], tracks=[TECHNICAL, OVAL])

    assert format_report(first) == format_report(second)
    assert markdown(first) == markdown(second)
    assert report_of([agents[0]], seed=1)["seeds"] != first["seeds"]


def test_a_report_says_how_each_agent_did_on_each_track(agents: tuple[Path, Path]) -> None:
    report = report_of(list(agents), tracks=[TECHNICAL, OVAL])

    assert (report["schema_version"], report["episodes"], report["start"]) == (1, 4, "random")
    assert len(report["seeds"]) == 4
    a, b = report["agents"]
    assert (a["label"], b["label"]) == ("A", "B")
    assert a["path"] == agents[0].as_posix()
    assert a["model_sha256"] == hashlib.sha256((agents[0] / "model.zip").read_bytes()).hexdigest()
    assert (a["algorithm"], a["steps"]) == ("PPO", 128)
    assert a["run"].endswith("seed0")
    assert a["settings"]["episode"]["time_limit"] == 3.0
    assert [track["path"] for track in report["tracks"]] == [TECHNICAL.as_posix(), OVAL.as_posix()]
    technical = report["tracks"][0]
    assert technical["name"] == "Technical Circuit"
    assert technical["sha256"] == hashlib.sha256(TECHNICAL.read_bytes()).hexdigest()
    for label in ("A", "B"):
        result = technical["results"][label]
        assert result["summary"]["runs"] == len(result["runs"]) == 4
        assert set(result["summary"]) >= {"completion_rate", "mean_lap", "best_lap", "off_tracks"}
        assert {"lap_times", "off_tracks", "clean"} <= set(result["runs"][0])


def test_each_agent_races_with_its_own_settings(agents: tuple[Path, Path]) -> None:
    report = report_of(list(agents))

    results = report["tracks"][0]["results"]
    assert max(run["seconds"] for run in results["A"]["runs"]) <= 3.0
    assert max(run["seconds"] for run in results["B"]["runs"]) <= 4.0


def test_the_evaluation_can_change_the_agents_settings(agents: tuple[Path, Path]) -> None:
    report = report_of(list(agents), overrides=["episode.time_limit=1.5"])

    assert report["overrides"] == ["episode.time_limit=1.5"]
    for result in report["tracks"][0]["results"].values():
        assert max(run["seconds"] for run in result["runs"]) <= 1.5
    assert "Changed for this evaluation: `episode.time_limit=1.5`." in markdown(report)


def test_a_change_to_what_agents_see_is_refused(agents: tuple[Path, Path]) -> None:
    with pytest.raises(IncompatibleModelError, match="rays"):
        report_of([agents[0]], overrides=["observation.rays=false"])


def test_start_places_differ_from_the_ones_training_tests_on(agents: tuple[Path, Path]) -> None:
    report = report_of([agents[0]])

    meta = json.loads((agents[0].parents[1] / "meta.json").read_text(encoding="utf-8"))
    assert not set(report["seeds"]) & set(meta["seeds"]["tests"])


def test_from_the_grid_every_run_is_the_same(agents: tuple[Path, Path]) -> None:
    report = report_of([agents[0]], start="grid")

    runs = report["tracks"][0]["results"]["A"]["runs"]
    assert all(run == runs[0] for run in runs)


def test_a_report_is_written_as_json(agents: tuple[Path, Path], tmp_path: Path) -> None:
    report = report_of([agents[0]])

    write_report(report, tmp_path / "reports" / "a.json")

    assert (tmp_path / "reports" / "a.json").read_text(encoding="utf-8") == format_report(report)


# --------------------------------------------------------------------------- #
# Side by side, in Markdown
# --------------------------------------------------------------------------- #


def summary(**changes: Any) -> dict[str, Any]:
    values = {
        "runs": 20,
        "completion_rate": 0.95,
        "laps": 38,
        "mean_lap": 30.912,
        "best_lap": 30.17,
        "off_tracks": 1,
        "average_speed": 31.24,
        "distance": 1989.4,
        "score": 198.861,
    }
    return values | changes


def agent(label: str, steps: int | None, time_limit: float) -> dict[str, Any]:
    return {
        "label": label,
        "path": f"runs/{label.lower()}/checkpoints/best",
        "algorithm": "PPO",
        "run": None if steps is None else f"run-{label.lower()}",
        "steps": steps,
        "settings": {"episode": {"time_limit": time_limit}, "training": {"seed": len(label)}},
    }


def test_agents_are_laid_out_side_by_side() -> None:
    report = {
        "episodes": 20,
        "seed": 0,
        "start": "random",
        "overrides": [],
        "agents": [agent("A", 301_056, 60.0), agent("B", None, 90.0)],
        "tracks": [
            {
                "path": "tracks/technical.json",
                "name": "Technical Circuit",
                "results": {
                    "A": {"summary": summary()},
                    "B": {
                        "summary": summary(
                            completion_rate=0.0, laps=0, mean_lap=None, best_lap=None, off_tracks=20
                        )
                    },
                },
            }
        ],
    }

    assert markdown(report) == (
        "# Evaluation\n"
        "\n"
        "20 runs per agent and track, from random places (seed 0), each agent with the settings "
        "it was trained with.\n"
        "\n"
        "The agents' settings differ:\n"
        "- `episode.time_limit`: A 60.0, B 90.0\n"
        "\n"
        "| Agent | Model                   | Algorithm | Run   |   Steps |\n"
        "| ----- | ----------------------- | --------- | ----- | ------: |\n"
        "| A     | runs/a/checkpoints/best | PPO       | run-a | 301,056 |\n"
        "| B     | runs/b/checkpoints/best | PPO       | -     |       - |\n"
        "\n"
        "## Technical Circuit (tracks/technical.json)\n"
        "\n"
        "|                 |           A |         B |\n"
        "| --------------- | ----------: | --------: |\n"
        "| Completion rate | 95% (19/20) | 0% (0/20) |\n"
        "| Laps            |          38 |         0 |\n"
        "| Mean lap        |     30.91 s |         - |\n"
        "| Best lap        |     30.17 s |         - |\n"
        "| Left the road   |           1 |        20 |\n"
        "| Average speed   |    31.2 m/s |  31.2 m/s |\n"
        "| Distance        |     1,989 m |   1,989 m |\n"
        "| Score           |      198.86 |    198.86 |\n"
    )


def test_the_text_says_where_the_runs_start() -> None:
    report = {
        "episodes": 1,
        "seed": 3,
        "start": "grid",
        "overrides": [],
        "agents": [agent("A", 1, 60.0)],
        "tracks": [],
    }

    assert "1 run per agent and track, from the grid (seed 3)" in markdown(report)


# --------------------------------------------------------------------------- #
# What can't be evaluated
# --------------------------------------------------------------------------- #


def test_tracks_are_found_by_name_or_pattern_each_once(tmp_path: Path) -> None:
    for name in ("b.json", "a.json", "notes.md"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    pattern = (tmp_path / "*.json").as_posix()

    found = find_tracks([pattern, str(tmp_path / "a.json")])

    assert found == [tmp_path / "a.json", tmp_path / "b.json"]


@pytest.mark.parametrize("pattern", ["missing.json", "nowhere/*.json", "."])
def test_a_pattern_without_track_files_is_refused(pattern: str) -> None:
    with pytest.raises(EvaluationError, match="no track file there"):
        find_tracks([pattern])


def test_a_track_with_errors_is_refused(agents: tuple[Path, Path], tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    write_track_file(
        TrackFile.from_arrays("Broken", [[0, 0], [10, 0], [5, 8]], [12, 12, 12]), broken
    )

    with pytest.raises(EvaluationError, match="the track has errors; `racecar check"):
        report_of([agents[0]], tracks=[broken])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"episodes": 0}, "episodes must be at least 1, got 0"),
        ({"start": "pit"}, "start must be 'random' or 'grid', got 'pit'"),
    ],
)
def test_settings_of_the_evaluation_are_checked(
    agents: tuple[Path, Path], options: dict[str, Any], message: str
) -> None:
    entrants = load_entrants([agents[0]])
    with pytest.raises(EvaluationError, match=message):
        evaluate(entrants, [TECHNICAL], **({"episodes": 4} | options))


def test_there_must_be_between_1_and_26_agents(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="give at least one agent"):
        load_entrants([])
    with pytest.raises(EvaluationError, match="at most 26 agents"):
        load_entrants([tmp_path] * 27)


def test_a_folder_without_a_saved_agent_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ModelCardError, match="can't read the model card"):
        load_entrants([tmp_path])


def test_an_invalid_change_names_where_it_came_from(agents: tuple[Path, Path]) -> None:
    with pytest.raises(ConfigError, match=r"episode.time_limit: .* \(from --set\)"):
        load_entrants([agents[0]], ["episode.time_limit=-1"])
