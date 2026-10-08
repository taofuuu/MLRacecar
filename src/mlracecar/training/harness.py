"""The evaluation harness: score agents on tracks and report how they did (`racecar eval`).

Each agent drives `episodes` runs on each track without learning, all at once
(`evaluation.drive_test_runs`), in the settings it was trained with (from its model card),
changed by any overrides. The runs start from random places set by one seed (or from the grid):
the same places for every agent, so agents are compared on the same runs, and different from
the places training tests on, so an agent picked as the best during training isn't flattered by
the places it was picked on.

With somewhere to record them, every run is also saved as a replay (`mlracecar.io.replay`),
named after its track, agent, and number, such as ``technical-A-03.npz``: exactly the run that
was scored, to watch with ``racecar replay``.

A report says how each agent did on each track: the completion rate (the share of clean runs:
until the time limit, never leaving the road), the laps and their mean and best times, the times
off the road, the average speed and distance, and the score; and every run. It holds nothing
else that could change between two evaluations (no dates, no timings), so the same agents,
tracks, and settings give the same report, byte for byte. `markdown` lays it out as tables with
one column per agent, side by side.
"""

import glob
import hashlib
import json
import string
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlracecar.agents.sb3 import CARD_FILE, MODEL_FILE, SB3Agent, check_compatible
from mlracecar.config.files import load_config
from mlracecar.config.models import RacecarConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.validation import has_errors, validate
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.io.model_card import read_model_card
from mlracecar.io.replay import Replay, write_replay
from mlracecar.io.track_file import TrackFile, read_track_file
from mlracecar.training.evaluation import alone, as_dict, drive_test_runs, summarize

SCHEMA_VERSION = 1
"""The report's format version."""

EVALUATION_STREAM = 2
"""Which random stream the start places come from: training tests use 1, so the places differ."""

NOT_FOR_RACING = ("training", "ppo")
"""Settings sections about how an agent learned, not how it races: left out of comparisons."""

type Report = dict[str, Any]
"""A report: plain JSON-ready data (see the module's description and `evaluate`)."""


class EvaluationError(ValueError):
    """An evaluation that can't be done, with the reason in plain words."""


@dataclass(frozen=True)
class Entrant:
    """An agent to evaluate: where it was saved, and the settings it races with."""

    label: str
    """Its name in the report: ``A``, ``B``, ..."""
    path: Path
    """Its folder."""
    agent: SB3Agent
    config: RacecarConfig
    """The settings it was trained with, changed by the evaluation's overrides."""


def load_entrants(paths: Sequence[str | Path], overrides: Sequence[str] = ()) -> list[Entrant]:
    """Load the agents saved in ``paths``, labelled ``A``, ``B``, ... in order.

    Args:
        paths: Saved agents' folders (`SB3Agent.save`).
        overrides: ``section.key=value`` changes to each agent's own settings.

    Raises:
        EvaluationError: Without agents, or with more than 26.
        ModelCardError: If an agent's card can't be read.
        ConfigError: If an override isn't valid, or the card's settings aren't.
    """
    if not paths:
        raise EvaluationError("nothing to evaluate: give at least one agent")
    if len(paths) > len(string.ascii_uppercase):
        raise EvaluationError(f"at most {len(string.ascii_uppercase)} agents at once")
    entrants = []
    for label, path in zip(string.ascii_uppercase, paths, strict=False):
        folder = Path(path)
        card = read_model_card(folder / CARD_FILE)
        config = load_config(overrides=overrides, base=(str(folder / CARD_FILE), card.config))
        entrants.append(Entrant(label, folder, SB3Agent.load(folder), config))
    return entrants


def find_tracks(patterns: Iterable[str]) -> list[Path]:
    """The track files that ``patterns`` name, each a path or a pattern such as
    ``tracks/*.json``, in order, each once.

    Raises:
        EvaluationError: If a pattern matches no file.
    """
    found: dict[Path, None] = {}
    for pattern in patterns:
        # glob.glob, not Path.glob: it takes absolute patterns too
        wild = any(character in pattern for character in "*?[")
        matches = sorted(glob.glob(pattern)) if wild else [pattern]  # noqa: PTH207
        matches = [match for match in matches if Path(match).is_file()]
        if not matches:
            raise EvaluationError(f"{pattern}: no track file there")
        found |= dict.fromkeys(Path(match) for match in matches)
    return list(found)


def evaluate(
    entrants: Sequence[Entrant],
    tracks: Sequence[Path],
    episodes: int = 20,
    seed: int = 0,
    start: str = "random",
    overrides: Sequence[str] = (),
    progress: Callable[[str], None] = lambda line: None,
    record: Path | None = None,
) -> Report:
    """Let every agent drive ``episodes`` runs on every track, and report how each went.

    Args:
        entrants: The agents (`load_entrants`).
        tracks: The track files.
        episodes: Runs per agent and track.
        seed: Sets the start places.
        start: ``"random"`` places on the lap, or ``"grid"`` (the same run every time, for an
            agent that drives the same way from the same place).
        overrides: The changes made to the agents' settings, to record in the report.
        progress: Told what's being driven, a line at a time.
        record: A folder to save every run in as a replay, or ``None``.

    Raises:
        EvaluationError: If ``episodes`` is under 1, ``start`` is unknown, or a track has
            errors.
        TrackFileError: If a track file can't be read.
        IncompatibleModelError: If an agent can't drive with its settings as changed.
    """
    if episodes < 1:
        raise EvaluationError(f"episodes must be at least 1, got {episodes}")
    if start not in ("random", "grid"):
        raise EvaluationError(f"start must be 'random' or 'grid', got {start!r}")
    seeds = [
        int(value)
        for value in np.random.SeedSequence([seed, EVALUATION_STREAM]).generate_state(episodes)
    ]
    report: Report = {
        "schema_version": SCHEMA_VERSION,
        "episodes": episodes,
        "seed": seed,
        "start": start,
        "seeds": seeds,
        "overrides": list(overrides),
        "agents": [_describe(entrant) for entrant in entrants],
        "tracks": [],
    }
    for path in tracks:
        track_file = _load_track(path)
        track = track_file.to_track()
        results: dict[str, Any] = {}
        for entrant, agent in zip(entrants, report["agents"], strict=True):
            progress(f"{entrant.label} on {path.as_posix()}: {episodes} runs")
            spec = BatchedRacingEnv(track, 1, entrant.config).observations.spec
            check_compatible(entrant.agent.card, spec, source=str(entrant.path))
            frames: list[Snapshot] = []
            runs = drive_test_runs(
                entrant.agent,
                track,
                entrant.config,
                seeds,
                start,
                on_step=None if record is None else frames.append,
            )
            results[entrant.label] = {
                "summary": summarize(runs),
                "runs": [as_dict(run) for run in runs],
            }
            if record is not None:
                dt = entrant.config.simulation.to_timing().decision_dt
                for index, run in enumerate(runs):
                    steps = round(run.seconds / dt)
                    info = {
                        "agent": {key: value for key, value in agent.items() if key != "settings"},
                        "track": {"path": path.as_posix(), "name": track_file.name},
                        "run": index + 1,
                        "seed": seeds[index],
                        "start": start,
                        "result": as_dict(run),
                    }
                    replay = Replay(
                        [alone(frame, index) for frame in frames[: steps + 1]],
                        track_file,
                        agent["settings"],
                        info,
                    )
                    write_replay(
                        replay, record / f"{path.stem}-{entrant.label}-{index + 1:02d}.npz"
                    )
        report["tracks"].append(
            {
                "path": path.as_posix(),
                "name": track_file.name,
                "sha256": _sha256(path),
                "results": results,
            }
        )
    return report


def format_report(report: Report) -> str:
    """The report as JSON text, the same for the same report."""
    return json.dumps(report, indent=2) + "\n"


def write_report(report: Report, path: str | Path) -> None:
    """Write the report as JSON, making the folder if needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_report(report), encoding="utf-8", newline="\n")


def markdown(report: Report) -> str:
    """The report as Markdown: the agents, then a table per track with a column per agent."""
    agents = report["agents"]
    labels = [agent["label"] for agent in agents]
    where = "from the grid" if report["start"] == "grid" else "from random places"
    lines = [
        "# Evaluation",
        "",
        f"{report['episodes']} run{'s' if report['episodes'] != 1 else ''} per agent and track, "
        f"{where} (seed {report['seed']}), each "
        "agent with the settings it was trained with.",
    ]
    if report["overrides"]:
        changes = ", ".join(f"`{override}`" for override in report["overrides"])
        lines.append(f"Changed for this evaluation: {changes}.")
    if differences := _differences([agent["settings"] for agent in agents]):
        lines.append("")
        lines.append("The agents' settings differ:")
        for key, values in differences.items():
            each = ", ".join(
                f"{label} {value}" for label, value in zip(labels, values, strict=True)
            )
            lines.append(f"- `{key}`: {each}")
    lines += [
        "",
        *_table(
            ["Agent", "Model", "Algorithm", "Run", "Steps"],
            [
                [
                    agent["label"],
                    agent["path"],
                    agent["algorithm"],
                    agent["run"] or "-",
                    "-" if agent["steps"] is None else f"{agent['steps']:,}",
                ]
                for agent in agents
            ],
            right=(4,),
        ),
    ]
    for track in report["tracks"]:
        summaries = [track["results"][label]["summary"] for label in labels]
        lines += ["", f"## {track['name']} ({track['path']})", ""]
        rows = [[name, *(show(summary) for summary in summaries)] for name, show in _ROWS]
        lines += _table(["", *labels], rows, right=tuple(range(1, len(labels) + 1)))
    return "\n".join(lines) + "\n"


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f} s"


_ROWS: tuple[tuple[str, Callable[[Mapping[str, Any]], str]], ...] = (
    (
        "Completion rate",
        lambda s: (
            f"{s['completion_rate']:.0%} ({round(s['completion_rate'] * s['runs'])}/{s['runs']})"
        ),
    ),
    ("Laps", lambda s: f"{s['laps']}"),
    ("Mean lap", lambda s: _seconds(s["mean_lap"])),
    ("Best lap", lambda s: _seconds(s["best_lap"])),
    ("Left the road", lambda s: f"{s['off_tracks']}"),
    ("Average speed", lambda s: f"{s['average_speed']:.1f} m/s"),
    ("Distance", lambda s: f"{s['distance']:,.0f} m"),
    ("Score", lambda s: f"{s['score']:.2f}"),
)
"""The rows of a track's table: a name, and how to show it from a summary."""


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], right: Sequence[int]) -> list[str]:
    """A Markdown table, padded so it lines up as plain text too."""
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]
    widths = [max(width, 3) for width in widths]

    def line(cells: Sequence[str]) -> str:
        padded = [
            cell.rjust(width) if column in right else cell.ljust(width)
            for column, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = [
        ("-" * (width - 1) + ":") if column in right else "-" * width
        for column, width in enumerate(widths)
    ]
    return [line(header), "| " + " | ".join(rule) + " |", *[line(row) for row in rows]]


def _describe(entrant: Entrant) -> dict[str, Any]:
    card = entrant.agent.card
    steps = card.metadata.get("steps")
    run = card.metadata.get("run")
    return {
        "label": entrant.label,
        "path": entrant.path.as_posix(),
        "model_sha256": _sha256(entrant.path / MODEL_FILE),
        "algorithm": card.algorithm,
        "run": run if isinstance(run, str) else None,
        "steps": steps if isinstance(steps, int) else None,
        "settings": entrant.config.model_dump(mode="json"),
    }


def _differences(settings: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """The racing settings whose values differ between agents, by dotted name."""
    flat = [_flatten(each) for each in settings]
    keys = sorted({key for each in flat for key in each})
    return {
        key: [each.get(key) for each in flat]
        for key in keys
        if key.split(".")[0] not in NOT_FOR_RACING
        and len({json.dumps(each.get(key)) for each in flat}) > 1
    }


def _flatten(settings: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in settings.items():
        if isinstance(value, Mapping):
            flat |= _flatten(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _load_track(path: Path) -> TrackFile:
    """The track file, if its track can be raced.

    Raises:
        TrackFileError: If the file can't be read.
        EvaluationError: If the track has errors.
    """
    track_file = read_track_file(path)
    if has_errors(validate(track_file.points, track_file.widths)):
        raise EvaluationError(f"{path}: the track has errors; `racecar check {path}` lists them")
    return track_file


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
