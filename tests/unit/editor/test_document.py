"""Tests for mlracecar.editor.document: opening and saving, and tracking unsaved changes."""

from pathlib import Path

import pytest

from mlracecar.editor.document import TrackDocument
from mlracecar.editor.draft import DEFAULT_NAME, TrackDraft
from mlracecar.io.track_file import TrackFileError, read_track_file

SAMPLES = Path(__file__).parents[3] / "tracks"
SQUARE = TrackDraft(
    points=((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0)),
    widths=(12.0,) * 4,
)


def test_opening_nothing_starts_a_new_untitled_track() -> None:
    document = TrackDocument.open()
    assert document.path is None
    assert document.saved == TrackDraft()


def test_opening_a_track_file() -> None:
    document = TrackDocument.open(SAMPLES / "oval.json")
    assert document.path == SAMPLES / "oval.json"
    assert document.saved.name == "Oval"
    assert len(document.saved.points) == 16


def test_opening_a_file_that_does_not_exist_yet_starts_a_track_named_after_it(
    tmp_path: Path,
) -> None:
    document = TrackDocument.open(tmp_path / "my-circuit.json")
    assert document.path == tmp_path / "my-circuit.json"
    assert document.saved == TrackDraft(name="my-circuit")


def test_opening_a_broken_file_raises(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("not json", encoding="utf-8")
    with pytest.raises(TrackFileError):
        TrackDocument.open(tmp_path / "broken.json")


def test_changes_are_unsaved_until_saved(tmp_path: Path) -> None:
    square = SQUARE.rename("Square")
    document = TrackDocument(tmp_path / "square.json", square)
    assert not document.is_modified(square)
    moved = square.move_point(0, (90.0, 90.0))
    assert document.is_modified(moved)
    assert document.save(moved) == tmp_path / "square.json"
    assert not document.is_modified(moved)
    assert read_track_file(tmp_path / "square.json").points.tolist()[0] == [90.0, 90.0]


def test_save_as_moves_the_document_to_the_new_file(tmp_path: Path) -> None:
    document = TrackDocument(tmp_path / "a.json", SQUARE)
    document.save(SQUARE, tmp_path / "b.json")
    assert document.path == tmp_path / "b.json"
    assert (tmp_path / "b.json").exists()
    assert not (tmp_path / "a.json").exists()


def test_saving_names_an_untitled_track_after_its_file(tmp_path: Path) -> None:
    document = TrackDocument.open()
    untitled = TrackDraft(points=SQUARE.points, widths=SQUARE.widths)
    assert untitled.name == DEFAULT_NAME
    document.save(untitled, tmp_path / "corner-test.json")
    assert document.saved.name == "corner-test"
    assert read_track_file(tmp_path / "corner-test.json").name == "corner-test"


def test_saving_rounds_to_the_centimetre(tmp_path: Path) -> None:
    document = TrackDocument(tmp_path / "square.json", SQUARE)
    rough = SQUARE.move_point(0, (90.123456, -0.001)).set_width(1, 12.378633)
    document.save(rough)
    assert document.saved.points[0] == (90.12, 0.0)
    assert document.saved.widths[1] == 12.38
    text = (tmp_path / "square.json").read_text(encoding="utf-8")
    assert '{"x": 90.12, "y": 0.0, "width": 12.0}' in text  # 0.0, not -0.0
    on_disk = TrackDraft.from_track_file(read_track_file(tmp_path / "square.json"))
    assert on_disk == document.saved
    assert document.is_modified(rough)  # the rough draft isn't what's on disk
    assert not document.is_modified(on_disk)


def test_saving_keeps_a_name_the_user_chose(tmp_path: Path) -> None:
    document = TrackDocument.open()
    document.save(SQUARE.rename("Square"), tmp_path / "square.json")
    assert document.saved.name == "Square"


def test_saving_creates_missing_folders(tmp_path: Path) -> None:
    TrackDocument.open().save(SQUARE, tmp_path / "new" / "folder" / "square.json")
    assert (tmp_path / "new" / "folder" / "square.json").exists()


def test_a_track_with_errors_can_be_saved(tmp_path: Path) -> None:
    narrow = SQUARE.set_width(1, 2.0)
    TrackDocument.open().save(narrow, tmp_path / "narrow.json")
    assert read_track_file(tmp_path / "narrow.json").widths.tolist()[1] == 2.0


def test_saving_needs_a_path() -> None:
    with pytest.raises(ValueError, match="choose where to save it"):
        TrackDocument.open().save(SQUARE)


def test_saving_needs_at_least_three_points(tmp_path: Path) -> None:
    document = TrackDocument.open()
    with pytest.raises(TrackFileError, match="at least 3 control points"):
        document.save(TrackDraft().append_point((0.0, 0.0)), tmp_path / "dot.json")
    assert document.path is None  # nothing changed


@pytest.mark.parametrize(
    ("name", "file"),
    [
        ("Untitled track", "untitled-track.json"),
        ("Monza (1971)!", "monza-1971.json"),
        ("???", "track.json"),
    ],
)
def test_new_tracks_are_offered_a_file_named_after_them(
    name: str, file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    document = TrackDocument.open()
    assert document.suggested_path(TrackDraft(name=name)) == Path(file)
    (tmp_path / "tracks").mkdir()
    assert document.suggested_path(TrackDraft(name=name)) == Path("tracks") / file


def test_a_saved_track_is_offered_its_own_file() -> None:
    document = TrackDocument.open(SAMPLES / "oval.json")
    assert document.suggested_path(document.saved) == SAMPLES / "oval.json"
