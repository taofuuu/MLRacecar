"""Videos from pictures: animated GIFs (Pillow) and MP4s (ffmpeg, from the `video` extra).

The pictures are read one at a time, so a long video never has to fit in memory at full size.
There's no pygame here, so training can write TensorBoard's videos with it too.
"""

import io
import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

type Frame = NDArray[np.uint8]
"""An RGB picture, ``(height, width, 3)``."""

FORMATS = (".gif", ".mp4")
"""The kinds of video `write_video` makes, by file extension."""

MP4_QUALITY = 6
"""ffmpeg's quality for MP4s, from 0 to 10 (H.264, constant rate factor 20): thin lines such as
the distance rays stay sharp, and a 640 x 400 lap is about 1.6 MB, a quarter of its GIF."""


class VideoError(ValueError):
    """A video that can't be made, with the reason in plain words."""


def write_video(path: str | Path, frames: Iterable[Frame], fps: float) -> None:
    """Make a video of ``frames``, shown ``fps`` a second: a GIF or an MP4, by the extension.

    GIFs loop, and their timing is in hundredths of a second, so rates that divide 100 (10, 20,
    25, 50) play exactly. An MP4's width and height must be even.

    Raises:
        VideoError: For another extension, no pictures, an MP4 of odd size, or MP4s without the
            ``video`` extra.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in FORMATS:
        raise VideoError(f"{target}: can't make that kind of video; give a .gif or .mp4 file")
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".gif":
        data, _ = gif(frames, fps)
        target.write_bytes(data)
    else:
        _write_mp4(target, iter(frames), fps)


def gif(
    frames: Iterable[Frame], fps: float, shrink: int = 1, skip: int = 1
) -> tuple[bytes, tuple[int, int]]:
    """An animated GIF of ``frames`` that loops, and its width and height.

    Args:
        frames: The pictures, in order, read one at a time.
        fps: Pictures a second.
        shrink: Make it this many times smaller (averaging each square of pixels).
        skip: Keep only every ``skip``-th picture (and show each that much longer).

    Every picture uses the first one's colours (up to 256), which is fast and keeps the file
    small; a colour that only appears later is drawn in the nearest of them.

    Raises:
        VideoError: If there are no pictures.
    """
    pictures = (_shrunk(frame, shrink) for frame in itertools.islice(frames, 0, None, skip))
    first = next(pictures, None)
    if first is None:
        raise VideoError("a video needs at least one picture")
    palette = first.quantize(colors=256, dither=Image.Dither.NONE)
    rest = (picture.quantize(palette=palette, dither=Image.Dither.NONE) for picture in pictures)
    buffer = io.BytesIO()
    palette.save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=round(1000 * skip / fps),
        loop=0,
    )
    return buffer.getvalue(), first.size


def _write_mp4(path: Path, frames: Iterator[Frame], fps: float) -> None:
    try:
        import imageio_ffmpeg  # only for MP4s, from the `video` extra
    except ImportError as error:
        raise VideoError(
            "MP4 videos need the video extra: uv sync --extra video "
            "(or pip install 'mlracecar[video]'); GIFs need nothing more"
        ) from error
    first = next(frames, None)
    if first is None:
        raise VideoError("a video needs at least one picture")
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        raise VideoError(f"an MP4's width and height must be even, not {width} x {height}")
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        quality=MP4_QUALITY,
        macro_block_size=2,
        ffmpeg_log_level="error",
    )
    writer.send(None)  # starts ffmpeg
    try:
        for frame in itertools.chain([first], frames):
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def _shrunk(frame: Frame, shrink: int) -> Image.Image:
    picture = Image.fromarray(np.ascontiguousarray(frame))
    return picture.reduce(shrink) if shrink > 1 else picture
