"""The `racecar` command line. Subcommands for training come later."""

import importlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from mlracecar import __version__
from mlracecar.config.files import ConfigError, format_config, load_config
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import has_errors, validate
from mlracecar.editor.document import TrackDocument
from mlracecar.io.replay import Replay, ReplayError, read_replay
from mlracecar.io.track_file import TrackFileError, read_track_file

if TYPE_CHECKING:
    from mlracecar.env.racing import ViewerFactory

app = typer.Typer(
    name="racecar",
    help="Design race tracks, train AI drivers, and race them.",
    no_args_is_help=True,
    add_completion=False,
)


LIBRARIES = {
    "numpy": "the simulation",
    "gymnasium": "the RL environment",
    "pydantic": "the settings",
    "pygame-ce": "drawing: racecar drive and edit (uv sync --extra render)",
    "torch": "training (uv sync --extra train, or --extra train-cpu without an NVIDIA GPU)",
    "stable-baselines3": "training",
    "tensorboard": "training charts",
}
"""The libraries `racecar doctor` reports on, and what each is for."""


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"mlracecar {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Design race tracks, train AI drivers, and race them."""


@app.command()
def check(
    track: Annotated[Path, typer.Argument(help="The track file to check, e.g. tracks/oval.json.")],
) -> None:
    """Check a track file: that it's well-formed, and that the track itself works.

    Exits with status 1 if the file can't be read or the track has errors (warnings are fine).
    """
    try:
        track_file = read_track_file(track)
    except TrackFileError as error:
        typer.echo(f"Can't open the track. {error}", err=True)
        raise typer.Exit(1) from None

    issues = validate(track_file.points, track_file.widths)
    try:
        length = f", {Track.build(track_file.points, track_file.widths).length:.0f} m"
    except ValueError:
        length = ""  # the points can't form a track; the issues below say why
    typer.echo(f'{track}: "{track_file.name}", {len(track_file.control_points)} points{length}')
    if not issues:
        typer.echo("No problems found.")
        return
    for issue in issues:
        typer.echo(f"  {issue.severity.value.upper()}: {issue.message}")
    if has_errors(issues):
        raise typer.Exit(1)


@app.command()
def edit(
    track: Annotated[
        Path | None,
        typer.Argument(help="The track file to open. If it doesn't exist yet, starts a new track."),
    ] = None,
) -> None:
    """Open the track editor: draw a track with the mouse, see the road and any problems as you
    go, and save it (Ctrl+S).

    Needs the `render` extra (pygame). Press H in the editor for the controls.
    """
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # no pygame banner on start
    try:
        from mlracecar.editor.app import run_editor  # here, not at the top: pygame is optional
    except ModuleNotFoundError as error:
        if error.name != "pygame":
            raise
        typer.echo(
            "The editor needs pygame. Install it with: pip install 'mlracecar[render]'", err=True
        )
        raise typer.Exit(1) from None

    try:
        document = TrackDocument.open(track)
    except TrackFileError as error:
        typer.echo(f"Can't open the track. {error}", err=True)
        raise typer.Exit(1) from None
    run_editor(document)


@app.command()
def drive(
    track: Annotated[Path, typer.Argument(help="The track to drive, e.g. tracks/gp-circuit.json.")],
    files: Annotated[
        list[Path] | None,
        typer.Option(
            "--config", metavar="FILE", help="A settings file (YAML). Repeat to apply several."
        ),
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help="Change one setting, e.g. --set race.off_track=reset. Repeat to change several.",
        ),
    ] = None,
) -> None:
    """Drive a car round a track with the keyboard, with lap times.

    Arrow keys or WASD drive; R restarts. Needs the `render` extra (pygame).
    """
    try:
        config = load_config(files or (), overrides or ())
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    try:
        track_file = read_track_file(track)
    except TrackFileError as error:
        typer.echo(f"Can't open the track. {error}", err=True)
        raise typer.Exit(1) from None
    if has_errors(validate(track_file.points, track_file.widths)):
        typer.echo(
            f"{track} has problems that make it undrivable. See them with: racecar check {track}",
            err=True,
        )
        raise typer.Exit(1)

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        from mlracecar.play.drive import run_drive  # here, not at the top: pygame is optional
    except ModuleNotFoundError as error:
        if error.name != "pygame":
            raise
        typer.echo(
            "Driving needs pygame. Install it with: pip install 'mlracecar[render]'", err=True
        )
        raise typer.Exit(1) from None
    run_drive(track_file.to_track(), config, title=track_file.name)


@app.command(name="config")
def show_config(
    files: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Settings files (YAML), applied in order. Each only needs the settings it changes."
        ),
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help="Change one setting, e.g. --set vehicle.mass=1500. Repeat to change several.",
        ),
    ] = None,
) -> None:
    """Show the settings that would be used: the defaults, changed by the files (in order) and
    then by --set.

    Exits with status 1 if a file can't be read or a setting is invalid.
    """
    try:
        config = load_config(files or (), overrides or ())
    except ConfigError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    typer.echo(format_config(config), nl=False)


TRAINING_LIBRARIES = frozenset({"torch", "stable_baselines3", "tensorboard", "PIL"})


@app.command()
def train(
    files: Annotated[
        list[Path] | None,
        typer.Argument(help="Settings files (YAML), e.g. configs/smoke.yaml, applied in order."),
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help="Change one setting, e.g. --set training.steps=200000. Repeat for several.",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(help="A name for the run, in its folder's name (default: track and seed)."),
    ] = None,
    runs: Annotated[Path, typer.Option(help="Where run folders are made.")] = Path("runs"),
    resume: Annotated[
        Path | None,
        typer.Option(help="Carry on a stopped run instead: its folder, e.g. runs/2026-...-seed0."),
    ] = None,
) -> None:
    """Train an AI driver, saving everything about the run in a new folder under runs/.

    The settings say which track, how long, and how PPO learns (see configs/default.yaml).
    TensorBoard shows how it's going (uv run tensorboard --logdir runs), with videos if pygame
    is installed. Press Ctrl+C to stop: the agent is saved first, and --resume carries on. Needs
    the training libraries (uv sync --extra train). Exits with status 1 if the run can't start.
    """
    try:
        from mlracecar.training.run import TrainingError, TrainingRun  # only for training
    except ImportError as error:
        _need_training_libraries(error, "Training")
    try:
        if resume is not None:
            if files or overrides or name:
                typer.echo(
                    "--resume carries on a run with its own settings: give nothing else.", err=True
                )
                raise typer.Exit(1)
            run = TrainingRun.resume(resume)
        else:
            config = load_config(files or (), overrides or ())
            run = TrainingRun.start(config, runs, name)
    except (ConfigError, TrackFileError, TrainingError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Training in {run.directory}")
    typer.echo(f"Watch it in TensorBoard: uv run tensorboard --logdir {run.directory.parent}")
    viewer = _race_pictures()
    if viewer is None and run.config.training.video_every:
        typer.echo("No videos: drawing them needs pygame (uv sync --extra render).")
    run.train(report=typer.echo, viewer=viewer)


@app.command(name="eval")
def evaluate(
    models: Annotated[
        list[Path],
        typer.Option(
            "--model",
            metavar="FOLDER",
            help="A saved agent, e.g. runs/<run>/checkpoints/best. Repeat to compare several.",
        ),
    ],
    tracks: Annotated[
        list[str] | None,
        typer.Option(
            "--tracks",
            metavar="FILES",
            help='Track files, or a pattern in quotes, e.g. "tracks/*.json". Repeat for more. '
            "Default: the track each agent trained on.",
        ),
    ] = None,
    episodes: Annotated[int, typer.Option(min=1, help="Runs per agent and track.")] = 20,
    seed: Annotated[
        int, typer.Option(min=0, help="Sets the start places: the same seed, the same places.")
    ] = 0,
    start: Annotated[
        str, typer.Option(help="Where runs start: random (places on the lap) or grid.")
    ] = "random",
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help="Change one of the agents' settings, e.g. --set episode.time_limit=120.",
        ),
    ] = None,
    json_file: Annotated[
        Path | None, typer.Option("--json", help="Also save the report as JSON here.")
    ] = None,
    markdown_file: Annotated[
        Path | None, typer.Option("--markdown", help="Also save the Markdown report here.")
    ] = None,
    record: Annotated[
        Path | None,
        typer.Option(
            metavar="FOLDER", help="Save every run here as a replay, to watch: racecar replay."
        ),
    ] = None,
) -> None:
    """Score saved agents on tracks, from the same start places every time, and print a report.

    Each agent drives with the settings it was trained with (changed by any --set), without
    learning. Give several --model to compare them side by side on the same runs. The report
    (Markdown) gives each agent's completion rate (runs until the time limit without leaving
    the road), laps and lap times, times off the road, and speed, per track. The same agents and
    seed always give the same report. Needs the training libraries (uv sync --extra train).
    Exits with status 1 if the evaluation can't be done.
    """
    try:
        from mlracecar.agents.sb3 import IncompatibleModelError
        from mlracecar.io.model_card import ModelCardError
        from mlracecar.training import harness  # only for training
    except ImportError as error:
        _need_training_libraries(error, "Evaluating")
    changes = overrides or []
    try:
        entrants = harness.load_entrants(models, changes)
        names = tracks or [entrant.config.training.track for entrant in entrants]
        report = harness.evaluate(
            entrants,
            harness.find_tracks(names),
            episodes,
            seed,
            start,
            changes,
            progress=lambda line: typer.echo(line, err=True),
            record=record,
        )
    except (
        ConfigError,
        TrackFileError,
        ModelCardError,
        IncompatibleModelError,
        harness.EvaluationError,
    ) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    text = harness.markdown(report)
    typer.echo(text, nl=False)
    if json_file is not None:
        harness.write_report(report, json_file)
        typer.echo(f"Saved the report in {json_file}", err=True)
    if markdown_file is not None:
        markdown_file.parent.mkdir(parents=True, exist_ok=True)
        markdown_file.write_text(text, encoding="utf-8", newline="\n")
        typer.echo(f"Saved the Markdown in {markdown_file}", err=True)
    if record is not None:
        typer.echo(f"Saved every run as a replay in {record}", err=True)


@app.command()
def replay(
    file: Annotated[
        Path, typer.Argument(help="A replay, e.g. one that racecar eval --record saved.")
    ],
) -> None:
    """Watch a recorded race.

    Space plays or pauses, Left and Right skip a second, Up and Down change the speed (x0.25 to
    x4), Home and End jump to the start and the end, and clicking or dragging the bar along the
    bottom goes anywhere. C changes the camera and 1-4 the overlays, as in racecar drive. Needs
    the `render` extra (pygame). Exits with status 1 if the replay can't be read.
    """
    try:
        recorded = read_replay(file)
        config = load_config(base=(str(file), recorded.settings))
    except (ReplayError, ConfigError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        from mlracecar.play.replay import run_replay  # here, not at the top: pygame is optional
    except ModuleNotFoundError as error:
        if error.name != "pygame":
            raise
        typer.echo(
            "Watching replays needs pygame. Install it with: pip install 'mlracecar[render]'",
            err=True,
        )
        raise typer.Exit(1) from None
    run_replay(recorded, config, _replay_title(recorded))


def _replay_title(recorded: Replay) -> str:
    """The track's name, and who drove which run if the replay says: ``Oval - A, run 3``."""
    agent = recorded.info.get("agent")
    label = agent.get("label") if isinstance(agent, dict) else None
    run = recorded.info.get("run")
    who = ", ".join(part for part in (label, None if run is None else f"run {run}") if part)
    return f"{recorded.track.name} - {who}" if who else recorded.track.name


def _need_training_libraries(error: ImportError, doing: str) -> NoReturn:
    """Say what to install if a training library is missing; otherwise, raise ``error``."""
    if error.name not in TRAINING_LIBRARIES:
        raise error
    typer.echo(
        f"{doing} needs PyTorch and Stable-Baselines3: uv sync --extra train "
        "(or --extra train-cpu without an NVIDIA GPU).",
        err=True,
    )
    raise typer.Exit(1) from None


def _race_pictures() -> "ViewerFactory | None":
    """Draws the race as pictures, for videos; ``None`` without pygame."""
    try:
        from mlracecar.render.viewer import RaceViewer  # pygame
    except ImportError as error:
        if error.name != "pygame":
            raise
        return None
    return RaceViewer


@app.command()
def doctor() -> None:
    """Show what is installed, and whether training can use an NVIDIA GPU."""
    typer.echo(f"mlracecar {__version__}")
    typer.echo(f"Python {platform.python_version()} ({sys.executable})")
    width = max(len(name) for name in LIBRARIES)
    for name, purpose in LIBRARIES.items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        typer.echo(f"{name:<{width}}  {version:<15}  {purpose}")
    typer.echo(f"GPU: {_gpu()}")


def _gpu() -> str:
    """Whether PyTorch can train on a CUDA GPU, and which; or why not."""
    try:
        torch = importlib.import_module("torch")  # by name: it's only there for training
    except ImportError:
        return "unknown: PyTorch isn't installed"
    if torch.version.cuda is None:
        return "not used: this PyTorch is built for the CPU only (uv sync --extra train)"
    if not torch.cuda.is_available():
        return (
            f"none found: PyTorch is built for CUDA {torch.version.cuda}; check the NVIDIA driver"
        )
    properties = torch.cuda.get_device_properties(0)
    memory = properties.total_memory / 1024**3
    return f"{properties.name} ({memory:.1f} GB), CUDA {torch.version.cuda}"
