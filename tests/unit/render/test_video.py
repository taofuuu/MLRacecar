"""Tests for mlracecar.render.video: GIFs and MP4s from pictures."""

import io
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageSequence

from mlracecar.render.video import Frame, VideoError, gif, write_video

RED, GREEN, BLUE, WHITE = (220, 30, 30), (30, 200, 60), (40, 60, 230), (255, 255, 255)


def moving_square(count: int, size: tuple[int, int] = (40, 64)) -> Iterator[Frame]:
    """Pictures of a white square moving right across a green field, one at a time."""
    height, width = size
    for index in range(count):
        frame = np.full((height, width, 3), GREEN, dtype=np.uint8)
        frame[8:16, 4 * index : 4 * index + 8] = WHITE
        yield frame


def frames_of(data: bytes) -> list[Image.Image]:
    return [frame.convert("RGB") for frame in ImageSequence.Iterator(Image.open(io.BytesIO(data)))]


# --------------------------------------------------------------------------- #
# GIFs
# --------------------------------------------------------------------------- #


def test_a_gif_keeps_the_pictures() -> None:
    frames = list(moving_square(6))

    data, size = gif(iter(frames), fps=20)

    assert size == (64, 40)
    pictures = frames_of(data)
    assert len(pictures) == 6
    for picture, frame in zip(pictures, frames, strict=True):
        np.testing.assert_array_equal(np.asarray(picture), frame)
    assert Image.open(io.BytesIO(data)).info["duration"] == 50  # 20 a second


def test_a_gif_can_be_smaller_and_skip_pictures() -> None:
    data, size = gif(moving_square(10), fps=20, shrink=2, skip=2)

    assert size == (32, 20)
    assert len(frames_of(data)) == 5
    assert Image.open(io.BytesIO(data)).info["duration"] == 100


def test_a_colour_that_only_appears_later_is_drawn_in_the_nearest_one() -> None:
    first = np.full((8, 8, 3), GREEN, dtype=np.uint8)
    first[:4] = BLUE
    later = first.copy()
    later[:2, :2] = RED  # not in the first picture

    data, _ = gif(iter([first, later]), fps=20)

    picture = np.asarray(frames_of(data)[1])
    assert tuple(picture[0, 0]) in {BLUE, GREEN}
    np.testing.assert_array_equal(picture[4:], first[4:])


def test_a_video_needs_a_picture(tmp_path: Path) -> None:
    with pytest.raises(VideoError, match="at least one picture"):
        gif(iter([]), fps=20)
    with pytest.raises(VideoError, match="at least one picture"):
        write_video(tmp_path / "empty.mp4", iter([]), fps=20)


def test_a_gif_file_is_written(tmp_path: Path) -> None:
    write_video(tmp_path / "clips" / "square.GIF", moving_square(4), fps=25)

    pictures = frames_of((tmp_path / "clips" / "square.GIF").read_bytes())
    assert len(pictures) == 4


# --------------------------------------------------------------------------- #
# MP4s
# --------------------------------------------------------------------------- #


def test_an_mp4_keeps_every_picture_at_its_size(tmp_path: Path) -> None:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "square.mp4"

    write_video(path, moving_square(12), fps=25)

    # Decoded by ffmpeg in one go (its streaming reader leaves pipes open on Linux).
    decode = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path)]
    raw = subprocess.run(
        [*decode, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 40, 64, 3)  # 64 x 40 each
    assert len(frames) == 12
    assert imageio_ffmpeg.count_frames_and_secs(str(path)) == (12, pytest.approx(12 / 25))
    # Lossy, but close: the square is where it was drawn, on the green field.
    expected = list(moving_square(12))
    assert np.abs(frames[5].astype(int) - expected[5]).mean() < 4


def test_an_mp4_must_be_of_even_size(tmp_path: Path) -> None:
    pytest.importorskip("imageio_ffmpeg")

    with pytest.raises(VideoError, match="must be even, not 63 x 40"):
        write_video(tmp_path / "odd.mp4", moving_square(2, (40, 63)), fps=25)


def test_mp4s_without_the_video_extra_say_what_to_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # importing it fails

    with pytest.raises(VideoError, match="MP4 videos need the video extra: uv sync --extra video"):
        write_video(tmp_path / "square.mp4", moving_square(2), fps=25)


def test_other_kinds_of_video_are_refused(tmp_path: Path) -> None:
    with pytest.raises(VideoError, match=r"square\.avi: can't make that kind of video"):
        write_video(tmp_path / "square.avi", moving_square(2), fps=25)
