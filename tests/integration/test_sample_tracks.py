"""The sample tracks shipped in tracks/ must load, stay canonical, and pass every track check."""

from pathlib import Path

import pytest

from mlracecar.core.track.validation import validate
from mlracecar.io.track_file import format_track_file, read_track_file

TRACKS = sorted((Path(__file__).parents[2] / "tracks").glob("*.json"))


def test_the_samples_are_there() -> None:
    assert {path.stem for path in TRACKS} >= {"oval", "gp-circuit"}


@pytest.mark.parametrize("path", TRACKS, ids=[path.stem for path in TRACKS])
def test_sample_track_is_canonical_and_has_no_issues(path: Path) -> None:
    track_file = read_track_file(path)
    assert path.read_text(encoding="utf-8") == format_track_file(track_file)
    assert validate(track_file.points, track_file.widths) == []
