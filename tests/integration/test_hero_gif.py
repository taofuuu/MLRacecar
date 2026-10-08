"""The README's GIF: its replay still reads, and the one command that makes it still works.

The command is in docs/replays.md. Making the full GIF takes a few seconds, so that's a `slow`
test (``uv run pytest -m slow --no-cov``); checking the replay is quick.
"""

from pathlib import Path

import pytest
from PIL import Image, ImageSequence
from typer.testing import CliRunner

from mlracecar.cli import app
from mlracecar.io.replay import read_replay
from mlracecar.play.replay import lap_times

ROOT = Path(__file__).parents[2]
REPLAY = ROOT / "docs" / "media" / "hero-lap.npz"
COMMAND = (
    "uv run racecar replay docs/media/hero-lap.npz --export docs/media/hero.gif"
    " --lap 1 --camera overview --size 640x400 --rays"
)
"""How the README's GIF is made (docs/replays.md shows it on one line)."""


def test_the_readme_gif_replay_reads_and_has_its_lap() -> None:
    replay = read_replay(REPLAY)

    start, end = lap_times(replay, 1)

    assert replay.track.name == "Technical Circuit"
    assert end - start == pytest.approx(30.28, abs=0.01)


def test_the_command_is_the_one_documented() -> None:
    assert COMMAND in (ROOT / "docs" / "replays.md").read_text(encoding="utf-8")


@pytest.mark.slow
def test_the_readme_gif_is_made_by_one_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    arguments = COMMAND.split()[3:]  # after "uv run racecar"
    arguments[arguments.index("docs/media/hero.gif")] = str(tmp_path / "hero.gif")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    with Image.open(tmp_path / "hero.gif") as made:
        pictures = len(list(ImageSequence.Iterator(made)))
        assert (made.size, pictures) == ((640, 400), 758)  # 30.28 s at 25 a second, and the end
