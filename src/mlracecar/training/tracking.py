"""Experiment tracking: where a training run's numbers and videos go (architecture 4.10).

A `Tracker` takes named numbers at a step (car-steps trained), and videos. `TensorBoardTracker`
writes them in a folder for TensorBoard; another tracker (Weights & Biases, say) only needs the
same three methods. Stable-Baselines3's own numbers (its losses, frames per second, ...) reach a
tracker through `SB3Output`, so everything a run tracks goes the same way.
"""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from stable_baselines3.common.logger import KVWriter
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

from mlracecar.render.video import Frame, gif


class Tracker(Protocol):
    """Where a training run's numbers and videos go. Steps are car-steps trained."""

    def scalars(self, step: int, values: Mapping[str, float]) -> None:
        """Record numbers by name, such as ``test/score``, at ``step``."""
        ...

    def video(self, step: int, name: str, frames: Iterable[Frame], fps: float) -> None:
        """Record a video at ``step``: its pictures in order, shown ``fps`` a second."""
        ...

    def close(self) -> None:
        """Finish writing."""
        ...


class TensorBoardTracker:
    """Writes for TensorBoard: ``tensorboard --logdir <folder>`` shows it, while it's written.

    Numbers become charts. Videos become animated GIFs, in TensorBoard's Images tab, at half the
    size and half the frame rate to keep them small: a minute of racing is about 4 MB.

    Args:
        directory: The folder to write in. Writing in it again adds to what's there.
    """

    def __init__(self, directory: Path) -> None:
        self._writer = SummaryWriter(str(directory))

    def scalars(self, step: int, values: Mapping[str, float]) -> None:
        """Add each number to its chart."""
        for name, value in values.items():
            self._writer.add_scalar(name, value, step)
        self._writer.flush()

    def video(self, step: int, name: str, frames: Iterable[Frame], fps: float) -> None:
        """Add the video as an animated GIF, at half the size and frame rate."""
        data, (width, height) = gif(frames, fps, shrink=2, skip=2)
        image = Summary.Image(height=height, width=width, colorspace=3, encoded_image_string=data)
        summary = Summary(value=[Summary.Value(tag=name, image=image)])
        # As SummaryWriter.add_video does, without the extra library it needs to make the GIF.
        writer = self._writer._get_file_writer()  # type: ignore[no-untyped-call]  # no hints
        writer.add_summary(summary, step)
        self._writer.flush()

    def close(self) -> None:
        """Write what's left and close the files."""
        self._writer.close()


class SB3Output(KVWriter):
    """A Stable-Baselines3 logger output that passes its numbers to a tracker::

        model.set_logger(Logger(None, [SB3Output(tracker)]))

    Like Stable-Baselines3's own TensorBoard output, it leaves out what's marked as not for
    TensorBoard (counters such as the number of updates), and anything that isn't a number.
    """

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0
    ) -> None:
        """Pass the numbers recorded since the last write to the tracker."""
        values = {
            key: float(value)
            for key, value in key_values.items()
            if isinstance(value, int | float | np.integer | np.floating)
            and "tensorboard" not in (key_excluded.get(key) or ())
        }
        if values:
            self.tracker.scalars(step, values)

    def close(self) -> None:
        """Nothing to close: whoever made the tracker closes it."""
