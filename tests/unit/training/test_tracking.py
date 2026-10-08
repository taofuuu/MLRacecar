"""Tests for mlracecar.training.tracking: a run's numbers and videos, for TensorBoard."""

import io
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from PIL import Image, ImageSequence
from stable_baselines3.common.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mlracecar.training.tracking import Frame, SB3Output, TensorBoardTracker, Tracker, gif

RED, GREEN, BLUE, WHITE = (220, 30, 30), (30, 200, 60), (40, 60, 230), (255, 255, 255)


class Recorder:
    """Keeps everything it's given."""

    def __init__(self) -> None:
        self.scalars_given: list[tuple[int, dict[str, float]]] = []

    def scalars(self, step: int, values: Mapping[str, float]) -> None:
        self.scalars_given.append((step, dict(values)))

    def video(self, step: int, name: str, frames: Iterable[Frame], fps: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def moving_square(count: int, size: tuple[int, int] = (40, 64)) -> Iterator[Frame]:
    """Pictures of a white square moving right across a green field, one at a time."""
    height, width = size
    for index in range(count):
        frame = np.full((height, width, 3), GREEN, dtype=np.uint8)
        frame[8:16, 4 * index : 4 * index + 8] = WHITE
        yield frame


def frames_of(data: bytes) -> list[Image.Image]:
    return [frame.convert("RGB") for frame in ImageSequence.Iterator(Image.open(io.BytesIO(data)))]


def events(directory: Path) -> EventAccumulator:
    accumulator = EventAccumulator(str(directory))
    accumulator.Reload()
    return accumulator


def test_tensorboard_draws_each_number_on_its_chart(tmp_path: Path) -> None:
    tracker: Tracker = TensorBoardTracker(tmp_path)

    tracker.scalars(100, {"test/score": 1.5, "test/distance": 300.0})
    tracker.scalars(200, {"test/score": 4.0})
    tracker.close()

    board = events(tmp_path)
    score = board.Scalars("test/score")
    assert [(point.step, point.value) for point in score] == [(100, 1.5), (200, 4.0)]
    assert [point.value for point in board.Scalars("test/distance")] == [300.0]


def test_a_video_is_an_animated_gif_at_half_the_size_and_frame_rate(tmp_path: Path) -> None:
    tracker = TensorBoardTracker(tmp_path)

    tracker.video(512, "test/grid_run", moving_square(10), fps=20)
    tracker.close()

    [image] = events(tmp_path).Images("test/grid_run")
    assert (image.step, image.width, image.height) == (512, 32, 20)
    pictures = frames_of(image.encoded_image_string)
    assert len(pictures) == 5  # every second picture
    assert all(picture.size == (32, 20) for picture in pictures)
    assert Image.open(io.BytesIO(image.encoded_image_string)).info["duration"] == 100  # 10 fps


def test_a_gif_keeps_the_pictures(tmp_path: Path) -> None:
    frames = list(moving_square(6))

    data, size = gif(iter(frames), fps=20)

    assert size == (64, 40)
    pictures = frames_of(data)
    assert len(pictures) == 6
    for picture, frame in zip(pictures, frames, strict=True):
        np.testing.assert_array_equal(np.asarray(picture), frame)


def test_a_colour_that_only_appears_later_is_drawn_in_the_nearest_one() -> None:
    first = np.full((8, 8, 3), GREEN, dtype=np.uint8)
    first[:4] = BLUE
    later = first.copy()
    later[:2, :2] = RED  # not in the first picture

    data, _ = gif(iter([first, later]), fps=20)

    picture = np.asarray(frames_of(data)[1])
    assert tuple(picture[0, 0]) in {BLUE, GREEN}
    np.testing.assert_array_equal(picture[4:], first[4:])


def test_a_video_needs_a_picture() -> None:
    with pytest.raises(ValueError, match="at least one picture"):
        gif(iter([]), fps=20)


def test_stable_baselines3_numbers_reach_the_tracker() -> None:
    recorder = Recorder()
    logger = Logger(None, [SB3Output(recorder)])

    logger.record("train/loss", np.float32(0.5))
    logger.record("time/fps", 900)
    logger.record("train/n_updates", 3, exclude="tensorboard")  # a counter, not for charts
    logger.record("train/note", "text")
    logger.dump(step=2048)
    logger.dump(step=4096)  # nothing new: nothing passed on

    assert recorder.scalars_given == [(2048, {"train/loss": 0.5, "time/fps": 900.0})]
