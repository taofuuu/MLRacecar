"""Tests for mlracecar.io.track_file: lossless round trips, clear errors, and versioning."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import mlracecar.io.track_file as track_file_module
from mlracecar.io.track_file import (
    CURRENT_VERSION,
    TrackFile,
    TrackFileError,
    format_track_file,
    parse_track_file,
    read_track_file,
    track_file_schema,
    write_track_file,
)

SCHEMA_PATH = Path(__file__).parents[3] / "docs" / "schemas" / f"track-file-v{CURRENT_VERSION}.json"
TRIANGLE = [
    {"x": 0, "y": 0, "width": 10},
    {"x": 100, "y": 0, "width": 10},
    {"x": 50, "y": 80, "width": 12},
]

finite = st.floats(allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=1e-6, max_value=1e6)
json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | finite | st.text(),
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(st.text(max_size=6), inner, max_size=3)
    ),
    max_leaves=8,
)
control_points = st.lists(
    st.fixed_dictionaries({"x": finite, "y": finite, "width": positive}), min_size=3, max_size=8
)


@st.composite
def track_files(draw: st.DrawFn) -> TrackFile:
    return TrackFile.model_validate(
        {
            "schema_version": 1,
            "name": draw(st.text(min_size=1)),
            "author": draw(st.text()),
            "description": draw(st.text()),
            "control_points": draw(control_points),
            "metadata": draw(st.dictionaries(st.text(max_size=8), json_values, max_size=3)),
        }
    )


def file_text(**fields: Any) -> str:
    """A track file's JSON with the given fields replaced (or removed, with ``None``)."""
    data: dict[str, Any] = {"schema_version": 1, "name": "Triangle", "control_points": TRIANGLE}
    data.update(fields)
    return json.dumps({key: value for key, value in data.items() if value is not None})


def points_with(index: int, point: dict[str, Any]) -> list[dict[str, Any]]:
    """The triangle's points with one replaced."""
    return [point if i == index else original for i, original in enumerate(TRIANGLE)]


# --------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------- #


@given(track_files())
def test_save_then_load_is_lossless(track_file: TrackFile) -> None:
    text = format_track_file(track_file)
    assert parse_track_file(text) == track_file
    assert format_track_file(parse_track_file(text)) == text  # the formatting is stable too


def test_files_round_trip_on_disk_without_leftovers(tmp_path: Path) -> None:
    original = TrackFile.from_arrays(
        "Délice ✓", [[0, 0], [100, 0], [50, 80]], [10, 10, 12], author="Ann"
    )
    path = tmp_path / "track.json"
    write_track_file(original, path)
    assert read_track_file(path) == original
    assert path.read_text(encoding="utf-8") == format_track_file(original)
    assert [p.name for p in tmp_path.iterdir()] == ["track.json"]  # the temporary file is gone


def test_format_puts_one_control_point_on_each_line() -> None:
    text = format_track_file(parse_track_file(file_text()))
    assert '    {"x": 100.0, "y": 0.0, "width": 10.0},\n' in text
    assert text.endswith("}\n")


def test_arrays_in_and_out() -> None:
    track_file = TrackFile.from_arrays("Triangle", [[0, 0], [100, 0], [50, 80]], [10, 10, 12])
    np.testing.assert_array_equal(track_file.points, [[0, 0], [100, 0], [50, 80]])
    np.testing.assert_array_equal(track_file.widths, [10, 10, 12])
    assert track_file.to_track(spacing=1.0).length > 0


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"schema_version": 1,', r"not valid JSON \(line 1, column 22\)"),
        ("[1, 2, 3]", "expected a JSON object"),
        (file_text(schema_version=None), "schema_version: must be a whole number .* got None"),
        (file_text(schema_version=True), "got True"),
        (file_text(schema_version=0), "got 0"),
        (file_text(name=""), "name: String should have at least 1 character"),
        (file_text(colour="red"), "colour: Extra inputs are not permitted"),
        (
            file_text(control_points=TRIANGLE[:2]),
            "control_points: a track needs at least 3 control points, got 2",
        ),
        (
            file_text(control_points=points_with(2, {"x": 1, "y": 2, "width": 0})),
            r"control_points\.2\.width: Input should be greater than 0",
        ),
        (
            file_text(control_points=points_with(0, {"x": "1", "y": 0, "width": 1})),
            r"control_points\.0\.x: Input should be a valid number",
        ),
        (
            file_text(control_points=points_with(0, {"x": 0, "y": 0})),
            r"control_points\.0\.width: Field required",
        ),
        (
            file_text().replace('"x": 0,', '"x": NaN,', 1),
            r"control_points\.0\.x: Input should be a finite number",
        ),
    ],
)
def test_invalid_files_say_what_and_where(text: str, message: str) -> None:
    with pytest.raises(TrackFileError, match=message):
        parse_track_file(text, source="track.json")


def test_errors_name_the_file() -> None:
    with pytest.raises(TrackFileError, match=r"^track\.json: "):
        parse_track_file("{}", source="track.json")


def test_a_file_from_a_newer_version_asks_to_update() -> None:
    with pytest.raises(TrackFileError, match=r"format version 2, .* Update MLRacecar to open it"):
        parse_track_file(file_text(schema_version=CURRENT_VERSION + 1))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_files_may_be_utf8_or_utf16(tmp_path: Path, encoding: str) -> None:
    # Windows PowerShell 5.1 writes UTF-16 with `>` and `Out-File`.
    original = parse_track_file(file_text(name="Ström ±50"))
    path = tmp_path / "track.json"
    path.write_text(format_track_file(original), encoding=encoding)  # non-ASCII kept as is

    assert read_track_file(path) == original


def test_a_file_that_is_not_text_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "track.json"
    path.write_bytes(b'{"name": "\xff"}')

    with pytest.raises(
        TrackFileError, match=r"track\.json: not readable text \(invalid start byte"
    ):
        read_track_file(path)


def test_reading_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TrackFileError, match=r"missing\.json: can't read the file"):
        read_track_file(tmp_path / "missing.json")


def test_from_arrays_checks_its_input() -> None:
    with pytest.raises(TrackFileError, match="3 control points but 2 widths"):
        TrackFile.from_arrays("x", [[0, 0], [1, 0], [0, 1]], [1, 1])
    with pytest.raises(TrackFileError, match=r"control_points\.1\.width"):
        TrackFile.from_arrays("x", [[0, 0], [1, 0], [0, 1]], [1, -1, 1])


# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #


def test_older_files_are_upgraded_step_by_step(monkeypatch: pytest.MonkeyPatch) -> None:
    def rename(old: str, new: str, version: int) -> track_file_module.Migration:
        return lambda data: {
            **{k: v for k, v in data.items() if k != old},
            "schema_version": version,
            new: data[old],
        }

    # Pretend the format is at version 3, with upgrades 1 -> 2 -> 3 that rename a field.
    monkeypatch.setattr(track_file_module, "CURRENT_VERSION", 3)
    monkeypatch.setattr(
        track_file_module,
        "MIGRATIONS",
        {1: rename("name", "title", 2), 2: rename("title", "label", 3)},
    )
    upgraded = track_file_module._upgrade({"schema_version": 1, "name": "Old"}, "old.json")
    assert upgraded == {"schema_version": 3, "label": "Old"}


def test_a_missing_upgrade_step_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(track_file_module, "CURRENT_VERSION", 2)
    with pytest.raises(TrackFileError, match="no upgrade available from format version 1"):
        track_file_module._upgrade({"schema_version": 1}, "old.json")


def test_published_schema_is_up_to_date() -> None:
    assert SCHEMA_PATH.read_text(encoding="utf-8") == track_file_schema(), (
        "docs/schemas is out of date: run `uv run python scripts/export_track_schema.py`"
    )
