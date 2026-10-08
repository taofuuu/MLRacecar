"""Smoke tests for the `racecar` command line."""

import importlib.util
import re
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import mlracecar
from mlracecar.cli import app
from mlracecar.config.models import RacecarConfig
from mlracecar.core.track.model import Track
from mlracecar.editor.document import TrackDocument
from mlracecar.editor.draft import TrackDraft
from mlracecar.io.track_file import TrackFile, read_track_file, write_track_file

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


def test_check_reports_a_file_that_is_not_text(tmp_path: Path) -> None:
    path = tmp_path / "picture.json"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    result = runner.invoke(app, ["check", str(path)])
    assert result.exit_code == 1
    assert "Can't open the track." in result.output
    assert "not readable text" in result.output


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
def opened(monkeypatch: pytest.MonkeyPatch) -> list[TrackDocument]:
    """Stands in for the editor window and records the document it was opened with."""
    documents: list[TrackDocument] = []

    def fake_run_editor(document: TrackDocument | None = None) -> TrackDraft:
        assert document is not None
        documents.append(document)
        return document.saved

    monkeypatch.setattr("mlracecar.editor.app.run_editor", fake_run_editor)
    return documents


def test_edit_with_no_file_starts_a_new_track(opened: list[TrackDocument]) -> None:
    result = runner.invoke(app, ["edit"])
    assert result.exit_code == 0
    assert opened == [TrackDocument(None, TrackDraft())]


def test_edit_opens_a_track_file(opened: list[TrackDocument]) -> None:
    result = runner.invoke(app, ["edit", str(SAMPLES / "oval.json")])
    assert result.exit_code == 0
    assert opened[0].path == SAMPLES / "oval.json"
    assert opened[0].saved.name == "Oval"


def test_edit_a_file_that_does_not_exist_yet_starts_a_track_named_after_it(
    opened: list[TrackDocument], tmp_path: Path
) -> None:
    result = runner.invoke(app, ["edit", str(tmp_path / "my-circuit.json")])
    assert result.exit_code == 0
    assert opened == [TrackDocument(tmp_path / "my-circuit.json", TrackDraft(name="my-circuit"))]


def test_edit_reports_an_unreadable_file(opened: list[TrackDocument], tmp_path: Path) -> None:
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


# --------------------------------------------------------------------------- #
# racecar config
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "default.yaml"


def test_config_shows_the_default_settings() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert result.output == DEFAULT_CONFIG.read_text(encoding="utf-8")


def test_config_applies_files_then_set(tmp_path: Path) -> None:
    path = tmp_path / "heavy-car.yaml"
    path.write_text("vehicle:\n  mass: 1800\n  width: 1.9\n", encoding="utf-8")
    result = runner.invoke(app, ["config", str(path), "--set", "vehicle.mass=1500"])
    assert result.exit_code == 0
    assert "  mass: 1500 " in result.output
    assert "  width: 1.9 " in result.output


def test_config_lists_invalid_settings() -> None:
    result = runner.invoke(app, ["config", "--set", "vehicle.mass=-5"])
    assert result.exit_code == 1
    assert "vehicle.mass: must be greater than 0, got -5 (from --set)" in result.output


def test_config_reports_an_unreadable_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["config", str(tmp_path / "missing.yaml")])
    assert result.exit_code == 1
    assert "missing.yaml: can't read the file" in result.output


# --------------------------------------------------------------------------- #
# racecar drive
# --------------------------------------------------------------------------- #


@pytest.fixture
def driven(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Track, RacecarConfig, str]]:
    """Stands in for the driving window and records what it was opened with."""
    calls: list[tuple[Track, RacecarConfig, str]] = []

    def fake_run_drive(track: Track, config: RacecarConfig, title: str) -> None:
        calls.append((track, config, title))

    monkeypatch.setattr("mlracecar.play.drive.run_drive", fake_run_drive)
    return calls


def test_drive_opens_the_track_with_the_settings(
    driven: list[tuple[Track, RacecarConfig, str]], tmp_path: Path
) -> None:
    settings = tmp_path / "car.yaml"
    settings.write_text("vehicle:\n  mass: 1500\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "drive",
            str(SAMPLES / "oval.json"),
            "--config",
            str(settings),
            "--set",
            "race.off_track=reset",
        ],
    )
    assert result.exit_code == 0, result.output
    ((track, config, title),) = driven
    assert title == read_track_file(SAMPLES / "oval.json").name
    assert track.length > 0
    assert config.vehicle.mass == 1500.0
    assert config.race.off_track == "reset"


def test_drive_reports_an_unreadable_track(
    driven: list[tuple[Track, RacecarConfig, str]], tmp_path: Path
) -> None:
    result = runner.invoke(app, ["drive", str(tmp_path / "missing.json")])
    assert result.exit_code == 1
    assert "Can't open the track." in result.output
    assert driven == []


def test_drive_refuses_a_track_with_errors(
    driven: list[tuple[Track, RacecarConfig, str]], tmp_path: Path
) -> None:
    path = write_track(tmp_path, [12.0] * 11 + [4.0])  # too narrow at one point
    result = runner.invoke(app, ["drive", str(path)])
    assert result.exit_code == 1
    assert "has problems that make it undrivable" in plain(result.output)
    assert driven == []


def test_drive_reports_invalid_settings(driven: list[tuple[Track, RacecarConfig, str]]) -> None:
    result = runner.invoke(app, ["drive", str(SAMPLES / "oval.json"), "--set", "vehicle.mass=-1"])
    assert result.exit_code == 1
    assert "vehicle.mass: must be greater than 0" in result.output
    assert driven == []


def test_drive_without_pygame_explains_how_to_get_it(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in [
        "mlracecar.play.drive",
        "mlracecar.render.race",
        "mlracecar.render.hud",
        "mlracecar.render.drawing",
    ]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    monkeypatch.setitem(sys.modules, "pygame", None)  # makes `import pygame` fail
    result = runner.invoke(app, ["drive", str(SAMPLES / "oval.json")])
    assert result.exit_code == 1
    assert "pip install 'mlracecar[render]'" in result.output


def test_drive_does_not_hide_other_import_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlracecar.play.drive", None)
    result = runner.invoke(app, ["drive", str(SAMPLES / "oval.json")])
    assert isinstance(result.exception, ModuleNotFoundError)


# --------------------------------------------------------------------------- #
# racecar doctor
# --------------------------------------------------------------------------- #


def fake_torch(cuda: str | None, *, gpu: bool) -> SimpleNamespace:
    """Just enough of PyTorch for `racecar doctor`, whether or not the real one is installed."""
    card = SimpleNamespace(name="Test GPU", total_memory=8 * 1024**3)
    return SimpleNamespace(
        version=SimpleNamespace(cuda=cuda),
        cuda=SimpleNamespace(is_available=lambda: gpu, get_device_properties=lambda index: card),
    )


def doctor(monkeypatch: pytest.MonkeyPatch, torch: SimpleNamespace | None) -> list[str]:
    monkeypatch.setitem(sys.modules, "torch", torch)  # None: importing it fails
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    return result.output.splitlines()


def test_doctor_lists_the_libraries_and_their_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = doctor(monkeypatch, fake_torch("13.0", gpu=True))

    assert lines[0] == f"mlracecar {mlracecar.__version__}"
    assert lines[1].startswith(f"Python {sys.version.split()[0]}")
    assert any(
        re.match(rf"numpy +{re.escape(metadata.version('numpy'))} +the simulation", line)
        for line in lines
    )
    assert any(line.startswith("gymnasium ") for line in lines)


@pytest.mark.parametrize(
    ("torch", "expected"),
    [
        (fake_torch("13.0", gpu=True), "GPU: Test GPU (8.0 GB), CUDA 13.0"),
        (fake_torch(None, gpu=False), "GPU: not used: this PyTorch is built for the CPU only"),
        (
            fake_torch("13.0", gpu=False),
            "GPU: none found: PyTorch is built for CUDA 13.0; check the NVIDIA driver",
        ),
        (None, "GPU: unknown: PyTorch isn't installed"),
    ],
)
def test_doctor_says_whether_training_can_use_the_gpu(
    monkeypatch: pytest.MonkeyPatch, torch: SimpleNamespace | None, expected: str
) -> None:
    assert doctor(monkeypatch, torch)[-1].startswith(expected)


def test_doctor_says_how_to_install_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def version(name: str) -> str:
        if name in {"torch", "stable-baselines3", "tensorboard"}:
            raise metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr("mlracecar.cli.metadata.version", version)

    lines = doctor(monkeypatch, None)

    torch_line = next(line for line in lines if line.startswith("torch "))
    assert "not installed" in torch_line
    assert "uv sync --extra train" in torch_line


# --------------------------------------------------------------------------- #
# racecar train
# --------------------------------------------------------------------------- #

needs_training = pytest.mark.skipif(
    importlib.util.find_spec("stable_baselines3") is None,
    reason="needs the training libraries (uv sync --extra train)",
)

TINY_RUN = """\
training:
  track: {track}
  steps: 128
  cars: 2
  checkpoint_every: 64
  eval_every: 64
  eval_runs: 1
ppo:
  steps_per_car: 32
  batch_size: 64
  epochs: 1
  layers: 1
  layer_size: 8
episode:
  time_limit: 2
"""


def tiny_run(folder: Path) -> Path:
    path = folder / "tiny.yaml"
    path.write_text(
        TINY_RUN.format(track=SAMPLES.joinpath("oval.json").as_posix()), encoding="utf-8"
    )
    return path


def test_train_says_what_to_install_without_the_training_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "mlracecar.training.run", raising=False)
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)  # importing it fails

    result = runner.invoke(app, ["train"])

    assert result.exit_code == 1
    assert "Training needs PyTorch and Stable-Baselines3: uv sync --extra train" in result.output


def test_train_doesnt_hide_other_import_problems(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlracecar.training.run", None)

    result = runner.invoke(app, ["train"])

    assert isinstance(result.exception, ImportError)


@needs_training
def test_train_runs_and_says_where(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["train", str(tiny_run(tmp_path)), "--runs", str(tmp_path / "runs"), "--name", "cli"]
    )

    assert result.exit_code == 0, result.output
    [folder] = (tmp_path / "runs").iterdir()
    assert folder.name.endswith("_cli")
    assert result.output.startswith(f"Training in {folder}\n")
    assert "step  64/128  score" in result.output
    assert "Finished: 128 steps." in result.output


@needs_training
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--set", "training.cars=0"], "training.cars: must be at least 1"),
        (["--set", "training.track=missing.json"], "missing.json: can't read the file"),
        (["--set", "ppo.batch_size=48"], "ppo.batch_size (48) must divide"),
        (["--resume", "nowhere", "--name", "x"], "--resume carries on a run with its own settings"),
    ],
)
def test_train_explains_why_a_run_cant_start(
    tmp_path: Path, arguments: list[str], message: str
) -> None:
    result = runner.invoke(
        app, ["train", str(tiny_run(tmp_path)), "--runs", str(tmp_path / "runs"), *arguments]
    )

    assert result.exit_code == 1
    assert message in result.output


@needs_training
def test_train_resume_needs_a_run_folder(tmp_path: Path) -> None:
    result = runner.invoke(app, ["train", "--resume", str(tmp_path)])

    assert result.exit_code == 1
    assert f"{tmp_path}: not a training run" in result.output
