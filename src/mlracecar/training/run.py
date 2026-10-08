"""Training runs: what `racecar train` does (architecture section 4.10, ADR-0006).

A run trains a PPO agent (Stable-Baselines3) on many cars at once (`SB3VecEnv`) and keeps
everything needed to reproduce or check the result in one folder::

    runs/2026-10-08_153012_technical-seed0/
    ├── config.yaml              every setting, as resolved (defaults < files < --set)
    ├── meta.json                status, git commit, versions, seeds, track, hardware, timings
    ├── checkpoints/
    │   ├── step_000100000/      the agent every `training.checkpoint_every` steps
    │   ├── best/                the agent that scored best in testing
    │   └── last/                the latest agent: what --resume carries on from
    └── eval/evaluations.jsonl   one line per test: the score and how the runs went

Each saved agent is an `SB3Agent` folder: the model and its model card.

**Testing.** Every `training.eval_every` steps, and at the end, the agent drives
`training.eval_runs` runs without learning, from random places on the lap that are the same at
every test, so scores are comparable. The score is the runs' mean reward. One run from the grid
is reported too, for information. The best-scoring agent is kept in ``checkpoints/best``.

**Reproducible.** On the CPU, the same settings and seed give the same run: the same agent and
the same test results (a test checks it). Everything random is seeded from `training.seed`.

**Stopping and resuming.** Ctrl+C saves the agent to ``checkpoints/last`` and marks the run
interrupted; `TrainingRun.resume` carries on from there. A resumed run keeps learning, but the
cars start fresh runs, so it won't match an uninterrupted run exactly.
"""

import hashlib
import json
import os
import platform
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from mlracecar.agents.sb3 import SB3Agent, git_commit, library_versions, make_model_card
from mlracecar.config.files import format_config, load_config
from mlracecar.config.models import RacecarConfig
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import has_errors, validate
from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.io.model_card import ModelCard
from mlracecar.io.track_file import read_track_file
from mlracecar.training.evaluation import as_dict, drive_test_runs, summarize
from mlracecar.training.vec_env import SB3VecEnv

CONFIG_FILE = "config.yaml"
META_FILE = "meta.json"
CHECKPOINTS = "checkpoints"
EVALUATIONS = Path("eval") / "evaluations.jsonl"
BEST = "best"
LAST = "last"

type Report = Callable[[str], None]
"""Where a run says how it's going: a line at a time."""


class TrainingError(ValueError):
    """A run that can't start or carry on, with the reason in plain words."""


@dataclass(frozen=True)
class Outcome:
    """How a call to `TrainingRun.train` ended."""

    status: Literal["finished", "interrupted"]
    steps: int
    """Car-steps trained so far."""
    best_score: float | None
    """The best test score so far, or ``None`` before the first test."""


class TrainingRun:
    """One training run and its folder. Make one with `start`, or `resume` a stopped one."""

    def __init__(self, directory: Path, config: RacecarConfig, meta: dict[str, Any]) -> None:
        self.directory = directory
        self.config = config
        self.meta = meta

    @classmethod
    def start(
        cls,
        config: RacecarConfig,
        runs: str | os.PathLike[str] = "runs",
        name: str | None = None,
    ) -> Self:
        """Make a new run's folder under ``runs``, with its settings and what's known so far.

        The folder is named after the time and ``name`` (by default the track and the seed).

        Raises:
            TrainingError: If the settings don't work together, or the track can't be raced.
            TrackFileError: If the track file can't be read.
        """
        _check_batches(config)
        track_path = Path(config.training.track)
        _load_track(track_path)
        now = datetime.now(UTC)
        label = name or f"{track_path.stem}-seed{config.training.seed}"
        directory = Path(runs) / f"{now.astimezone():%Y-%m-%d_%H%M%S}_{_slug(label)}"
        directory.mkdir(parents=True)
        (directory / CONFIG_FILE).write_text(format_config(config), encoding="utf-8", newline="\n")
        training = config.training
        meta: dict[str, Any] = {
            "name": label,
            "status": "created",
            "created": now.isoformat(),
            "git": None if (commit := git_commit()) is None else commit.model_dump(),
            "versions": library_versions(),
            "track": {"path": track_path.as_posix(), "sha256": _sha256(track_path)},
            "seeds": {
                "training": training.seed,
                "cars": [training.seed + car for car in range(training.cars)],
                "tests": _test_seeds(training.seed, training.eval_runs),
            },
            "hardware": _hardware(training.device),
            "steps": 0,
            "best": None,
            "sessions": [],
        }
        run = cls(directory, config, meta)
        run._save_meta()
        return run

    @classmethod
    def resume(cls, directory: str | os.PathLike[str]) -> Self:
        """Open a stopped run to carry on training it.

        Raises:
            TrainingError: If it isn't a run folder, has nothing to resume from, is finished, or
                its track file has changed since.
        """
        folder = Path(directory)
        if not (folder / META_FILE).is_file() or not (folder / CONFIG_FILE).is_file():
            raise TrainingError(f"{folder}: not a training run (no {META_FILE} or {CONFIG_FILE})")
        meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
        if meta["status"] == "finished":
            raise TrainingError(f"{folder}: this run is finished; start a new one to train more")
        if not (folder / CHECKPOINTS / LAST).is_dir():
            raise TrainingError(f"{folder}: no saved agent to carry on from; start a new run")
        track = Path(meta["track"]["path"])
        if not track.is_file() or _sha256(track) != meta["track"]["sha256"]:
            raise TrainingError(
                f"{folder}: the track {track} has changed (or gone) since this run started, so "
                "carrying on would train on a different track; start a new run"
            )
        return cls(folder, load_config([folder / CONFIG_FILE]), meta)

    def train(self, report: Report = print) -> Outcome:
        """Train until `training.steps` car-steps, testing and saving along the way.

        Ctrl+C stops it cleanly: the agent is saved to ``checkpoints/last`` first.
        """
        training, ppo = self.config.training, self.config.ppo
        track = _load_track(Path(self.meta["track"]["path"]))
        env = SB3VecEnv(
            BatchedRacingEnv(
                track, training.cars, self.config, autoreset_mode=AutoresetMode.SAME_STEP
            )
        )
        last = self.directory / CHECKPOINTS / LAST / "model.zip"
        if last.is_file():
            model = PPO.load(last, env=env, device=training.device)
            # A fresh but repeatable stream of random numbers for the rest of the run.
            model.set_random_seed(training.seed + model.num_timesteps)
        else:
            set_random_seed(training.seed)
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=ppo.learning_rate,
                n_steps=ppo.steps_per_car,
                batch_size=ppo.batch_size,
                n_epochs=ppo.epochs,
                gamma=ppo.gamma,
                gae_lambda=ppo.gae_lambda,
                clip_range=ppo.clip_range,
                ent_coef=ppo.entropy_coef,
                policy_kwargs={"net_arch": [ppo.layer_size] * ppo.layers},
                device=training.device,
                seed=training.seed,
                verbose=0,
            )
        card = make_model_card(
            model, self.config, env.env.observations.spec, {"run": self.meta["name"]}
        )
        progress = _Progress(self, track, card, report)
        session: dict[str, Any] = {
            "started": datetime.now(UTC).isoformat(),
            "from_step": model.num_timesteps,
        }
        self.meta["sessions"].append(session)
        self.meta["status"] = "running"
        self._save_meta()
        status: Literal["finished", "interrupted"] = "finished"
        clock = time.perf_counter()
        try:
            remaining = max(training.steps - model.num_timesteps, 0)
            model.learn(remaining, callback=progress, reset_num_timesteps=False)
        except KeyboardInterrupt:
            status = "interrupted"
        steps = model.num_timesteps
        progress.save(model, LAST, steps)
        if status == "finished":
            if progress.tested_at != steps:
                progress.test(model, steps)
            if progress.saved_at != steps:
                progress.save(model, _step_folder(steps), steps)
        seconds = time.perf_counter() - clock
        session |= {
            "ended": datetime.now(UTC).isoformat(),
            "to_step": steps,
            "seconds": round(seconds, 3),
            "steps_per_second": round((steps - session["from_step"]) / max(seconds, 1e-9), 1),
        }
        self.meta |= {"status": status, "steps": steps}
        self._save_meta()
        best = self.meta["best"]
        if status == "interrupted":
            report(
                f"Stopped at step {steps:,}. Carry on with: racecar train --resume {self.directory}"
            )
        else:
            best_folder = self.directory / CHECKPOINTS / BEST
            report(f"Finished: {steps:,} steps. The best agent is in {best_folder}")
        return Outcome(status, steps, None if best is None else best["score"])

    def evaluations(self) -> list[dict[str, Any]]:
        """Every test so far, oldest first."""
        path = self.directory / EVALUATIONS
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _save_meta(self) -> None:
        path = self.directory / META_FILE
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(self.meta, indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)


class _Progress(BaseCallback):
    """Saves and tests the agent as training goes, and says how it's going."""

    def __init__(self, run: TrainingRun, track: Track, card: ModelCard, report: Report) -> None:
        super().__init__()
        self.run = run
        self.track = track
        self.card = card
        self.report = report
        self.saved_at = -1
        self.tested_at = -1
        self.clock = time.perf_counter()

    def _on_training_start(self) -> None:
        training = self.run.config.training
        steps = self.model.num_timesteps
        self.next_save = (steps // training.checkpoint_every + 1) * training.checkpoint_every
        self.next_test = (steps // training.eval_every + 1) * training.eval_every

    def _on_step(self) -> bool:
        training = self.run.config.training
        steps = self.num_timesteps
        if steps >= self.next_save:
            self.save(self.model, _step_folder(steps), steps)
            self.save(self.model, LAST, steps)
            self.next_save = (steps // training.checkpoint_every + 1) * training.checkpoint_every
        if steps >= self.next_test:
            self.test(self.model, steps)
            self.next_test = (steps // training.eval_every + 1) * training.eval_every
        return True

    def save(self, model: Any, folder: str, steps: int) -> None:
        """Save the agent, with its card, to ``checkpoints/<folder>``."""
        card = self.card.model_copy(
            update={"created": datetime.now(UTC), "metadata": self.card.metadata | {"steps": steps}}
        )
        SB3Agent(model, card).save(self.run.directory / CHECKPOINTS / folder)
        if folder != LAST:
            self.saved_at = steps

    def test(self, model: Any, steps: int) -> None:
        """Test the agent; keep it as the best if it scored best; say how it went."""
        run, config = self.run, self.run.config
        agent = SB3Agent(model, self.card)
        runs = drive_test_runs(agent, self.track, config, run.meta["seeds"]["tests"])
        grid = drive_test_runs(agent, self.track, config, [config.training.seed], start="grid")[0]
        summary = summarize(runs)
        record = {
            "step": steps,
            "seconds": round(time.perf_counter() - self.clock, 3),
            **summary,
            "grid": as_dict(grid),
        }
        path = run.directory / EVALUATIONS
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(record) + "\n")
        self.tested_at = steps
        best = run.meta["best"]
        improved = best is None or summary["score"] > best["score"]
        if improved:
            self.save(model, BEST, steps)
            run.meta["best"] = {"step": steps, "score": summary["score"]}
        run.meta["steps"] = steps
        run._save_meta()
        total = config.training.steps
        lap = "-" if summary["best_lap"] is None else f"{summary['best_lap']:.2f} s"
        self.report(
            f"step {steps:>{len(f'{total:,}')},}/{total:,}  score {summary['score']:8.2f}"
            f"{'  (best)' if improved else '        '}  {summary['distance']:6.0f} m"
            f"  laps {summary['laps']}  best lap {lap}  grid {grid.distance:.0f} m"
        )


def _check_batches(config: RacecarConfig) -> None:
    """PPO learns from batches of the experience each update gathers: they must divide it."""
    cars, per_car = config.training.cars, config.ppo.steps_per_car
    if (cars * per_car) % config.ppo.batch_size:
        raise TrainingError(
            f"ppo.batch_size ({config.ppo.batch_size}) must divide the experience each update "
            f"gathers: training.cars x ppo.steps_per_car = {cars} x {per_car} = {cars * per_car}"
        )


def _load_track(path: Path) -> Track:
    """The track in ``path``, if it can be raced.

    Raises:
        TrackFileError: If the file can't be read.
        TrainingError: If the track has errors.
    """
    track_file = read_track_file(path)
    if has_errors(validate(track_file.points, track_file.widths)):
        raise TrainingError(f"{path}: the track has errors; `racecar check {path}` lists them")
    return track_file.to_track()


def _test_seeds(seed: int, runs: int) -> list[int]:
    """The test runs' seeds: the same for every test of a run, and set by its seed."""
    return [int(value) for value in np.random.SeedSequence([seed, 1]).generate_state(runs)]


def _hardware(device: str) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else None
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "gpu": gpu,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "run"


def _step_folder(steps: int) -> str:
    return f"step_{steps:09d}"
