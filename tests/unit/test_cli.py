"""Smoke tests for the `racecar` command line."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import mlracecar
from mlracecar.cli import app
from mlracecar.editor.draft import TrackDraft
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


# --------------------------------------------------------------------------- #
# racecar edit
# --------------------------------------------------------------------------- #


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[TrackDraft]:
    """Stands in for the editor window and records the draft it was opened with."""
    drafts: list[TrackDraft] = []

    def fake_run_editor(draft: TrackDraft | None = None) -> TrackDraft:
        drafts.append(draft or TrackDraft())
        return drafts[-1]

    monkeypatch.setattr("mlracecar.editor.app.run_editor", fake_run_editor)
    return drafts


def test_edit_with_no_file_starts_a_new_track(opened: list[TrackDraft]) -> None:
    result = runner.invoke(app, ["edit"])
    assert result.exit_code == 0
    assert opened == [TrackDraft()]


def test_edit_opens_a_track_file(opened: list[TrackDraft]) -> None:
    result = runner.invoke(app, ["edit", str(SAMPLES / "oval.json")])
    assert result.exit_code == 0
    assert opened[0].name == "Oval"
    assert len(opened[0].points) == 16


def test_edit_a_file_that_does_not_exist_yet_starts_a_track_named_after_it(
    opened: list[TrackDraft], tmp_path: Path
) -> None:
    result = runner.invoke(app, ["edit", str(tmp_path / "my-circuit.json")])
    assert result.exit_code == 0
    assert opened == [TrackDraft(name="my-circuit")]


def test_edit_reports_an_unreadable_file(opened: list[TrackDraft], tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")
    result = runner.invoke(app, ["edit", str(path)])
    assert result.exit_code == 1
    assert "Can't open the track." in result.output
    assert opened == []


def test_edit_without_pygame_explains_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ["mlracecar.editor.app", "mlracecar.editor.view", "mlracecar.render.drawing"]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setitem(sys.modules, "pygame", None)  # makes `import pygame` fail
    result = runner.invoke(app, ["edit"])
    assert result.exit_code == 1
    assert "pip install 'mlracecar[render]'" in result.output


def test_edit_does_not_hide_other_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlracecar.editor.app", None)
    result = runner.invoke(app, ["edit"])
    assert isinstance(result.exception, ModuleNotFoundError)
