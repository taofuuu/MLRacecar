"""The `racecar` command line. Subcommands for training come later."""

import importlib
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer

from mlracecar import __version__
from mlracecar.config.files import ConfigError, format_config, load_config
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import has_errors, validate
from mlracecar.editor.document import TrackDocument
from mlracecar.io.track_file import TrackFileError, read_track_file

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
