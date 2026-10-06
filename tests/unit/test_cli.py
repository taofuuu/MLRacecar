"""Smoke tests for the `racecar` command line."""

import re
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import mlracecar
from mlracecar.cli import app
from mlracecar.io.track_file import TrackFile, write_track_file

runner = CliRunner()

# Typer colors its help output when it detects CI (e.g. GITHUB_ACTIONS is set). The color
# codes split words like "--version", so tests compare against the plain text.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"mlracecar {mlracecar.__version__}"


def test_version_follows_pep_440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", mlracecar.__version__)


def test_no_arguments_shows_help() -> None:
    output = plain(runner.invoke(app, []).output)

    assert "Usage" in output
    assert "--version" in output


# --------------------------------------------------------------------------- #
# racecar check
# --------------------------------------------------------------------------- #

SAMPLES = Path(__file__).parents[2] / "tracks"


def write_track(folder: Path, widths: list[float]) -> Path:
    angles = np.linspace(0, 2 * np.pi, len(widths), endpoint=False)
    points = np.column_stack([120 * np.cos(angles), 60 * np.sin(angles)])
    path = folder / "track.json"
    write_track_file(TrackFile.from_arrays("Test oval", points, widths), path)
    return path


def test_check_a_good_track() -> None:
    result = runner.invoke(app, ["check", str(SAMPLES / "oval.json")])
    assert result.exit_code == 0
    assert '"Oval", 16 points, 739 m' in result.output
    assert "No problems found." in result.output


def test_check_reports_an_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 1, "name": ""}', encoding="utf-8")
    result = runner.invoke(app, ["check", str(path)])
    assert result.exit_code == 1
    assert "Can't open the track." in result.output
    assert "name: String should have at least 1 character" in result.output


def test_check_fails_on_track_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["check", str(write_track(tmp_path, [12.0] * 11 + [4.0]))])
    assert result.exit_code == 1
    assert "ERROR: The road at point 11 is 4.0 m wide" in result.output


def test_check_passes_with_only_warnings(tmp_path: Path) -> None:
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    # A long thin oval: its ends are tighter than a car can steer, a warning but not an error.
    points = np.column_stack([120 * np.cos(angles), 24.5 * np.sin(angles)])
    thin = tmp_path / "thin.json"
    write_track_file(TrackFile.from_arrays("Thin oval", points, [6.0] * 16), thin)
    result = runner.invoke(app, ["check", str(thin)])
    assert result.exit_code == 0
    assert "WARNING: The bend" in result.output


def test_check_names_points_that_cannot_form_a_track(tmp_path: Path) -> None:
    path = tmp_path / "stacked.json"
    stacked = TrackFile.from_arrays("Stacked", [[0, 0], [0, 0], [50, 0], [0, 50]], [10.0] * 4)
    write_track_file(stacked, path)
    result = runner.invoke(app, ["check", str(path)])
    assert result.exit_code == 1
    assert '"Stacked", 4 points\n' in result.output  # no length: the points can't form a track
    assert "Point 0 is on top of point 1" in result.output
