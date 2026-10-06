"""The `racecar` command line. Subcommands for driving and training come later."""

import os
from pathlib import Path
from typing import Annotated

import typer

from mlracecar import __version__
from mlracecar.core.track.model import Track
from mlracecar.core.track.validation import has_errors, validate
from mlracecar.editor.draft import TrackDraft
from mlracecar.io.track_file import TrackFileError, read_track_file

app = typer.Typer(
    name="racecar",
    help="Design race tracks, train AI drivers, and race them.",
    no_args_is_help=True,
    add_completion=False,
)


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
    """Open the track editor: draw a track with the mouse and see the road as you go.

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

    if track is None:
        draft = TrackDraft()
    elif track.exists():
        try:
            draft = TrackDraft.from_track_file(read_track_file(track))
        except TrackFileError as error:
            typer.echo(f"Can't open the track. {error}", err=True)
            raise typer.Exit(1) from None
    else:
        draft = TrackDraft(name=track.stem)
    run_editor(draft)
