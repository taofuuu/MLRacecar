"""Track files: the versioned JSON format tracks are saved in (ADR-0004).

A file stores only what the user placed: the control points and the road width at each, plus a
name, author, description, and free-form ``metadata`` for tools (an editor's view settings, for
example). Everything else is derived again by `mlracecar.core.track` when the file is loaded.

Reading checks that a file is *well-formed*: valid JSON, a known format version, the right
fields with the right types, finite coordinates, positive widths, at least 3 points. Whether the
*track* is any good is a separate question for `mlracecar.core.track.validation`, so unfinished
tracks can still be saved and reopened.

Example (format version 1)::

    {
      "schema_version": 1,
      "name": "Oval",
      "author": "MLRacecar",
      "description": "",
      "control_points": [
        {"x": 120.0, "y": 0.0, "width": 12.0},
        {"x": 103.9, "y": 30.0, "width": 12.0},
        {"x": 60.0, "y": 52.0, "width": 12.0}
      ],
      "metadata": {}
    }

Files are written with one control point per line, so a diff shows exactly which point moved.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from pydantic_core import PydanticCustomError

from mlracecar.core.geometry import FloatArray
from mlracecar.core.track.model import Track

CURRENT_VERSION = 1
"""The format version this code writes, and the newest it can read."""

type Migration = Callable[[dict[str, Any]], dict[str, Any]]

MIGRATIONS: dict[int, Migration] = {}
"""Upgrades for older files: ``MIGRATIONS[n]`` turns a version-``n`` file into version ``n + 1``.

Empty while version 1 is the only version. When the format changes, bump `CURRENT_VERSION`,
add the upgrade from the previous version here, and keep the old example files in the tests.
"""


class TrackFileError(ValueError):
    """A track file can't be read. The message says which file and, if relevant, which field."""


class ControlPoint(BaseModel):
    """One point the user placed, with the road width there."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, allow_inf_nan=False, use_attribute_docstrings=True
    )

    x: float
    """Metres."""
    y: float
    """Metres."""
    width: Annotated[float, Field(gt=0)]
    """Road width at this point, in metres."""


class TrackFile(BaseModel):
    """The contents of a track file (format version 1)."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        title="MLRacecar track file",
        use_attribute_docstrings=True,
    )

    schema_version: Literal[1]
    """Format version; files from newer versions of MLRacecar are rejected with a clear message."""
    name: Annotated[str, Field(min_length=1)]
    author: str = ""
    description: str = ""
    # A JSON array arrives as a list; it's stored as a tuple so a loaded file can't be changed.
    control_points: Annotated[
        tuple[ControlPoint, ...], Field(strict=False, json_schema_extra={"minItems": 3})
    ]
    """The points in driving order; the first one is on the start/finish line."""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Free-form data for tools; MLRacecar itself ignores it."""

    @field_validator("control_points")
    @classmethod
    def _at_least_three_points(cls, points: tuple[ControlPoint, ...]) -> tuple[ControlPoint, ...]:
        # Counted only once every point is valid, so one bad point doesn't also read as "too few".
        if len(points) < 3:
            raise PydanticCustomError(
                "too_few_points",
                "a track needs at least 3 control points, got {count}",
                {"count": len(points)},
            )
        return points

    @classmethod
    def from_arrays(
        cls,
        name: str,
        points: ArrayLike,
        widths: ArrayLike,
        *,
        author: str = "",
        description: str = "",
        metadata: dict[str, JsonValue] | None = None,
    ) -> Self:
        """Build a track file from control points of shape ``(P, 2)`` and widths of shape ``(P,)``.

        Raises:
            TrackFileError: If the values don't make a well-formed track file.
        """
        xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        width = np.asarray(widths, dtype=np.float64).reshape(-1)
        if len(xy) != len(width):
            raise TrackFileError(f"{len(xy)} control points but {len(width)} widths")
        data = {
            "schema_version": CURRENT_VERSION,
            "name": name,
            "author": author,
            "description": description,
            "control_points": [
                {"x": float(x), "y": float(y), "width": float(w)}
                for (x, y), w in zip(xy, width, strict=True)
            ],
            "metadata": metadata or {},
        }
        return _validated(cls, data, source="track")

    @property
    def points(self) -> FloatArray:
        """Control points, shape ``(P, 2)``."""
        return np.array([[p.x, p.y] for p in self.control_points], dtype=np.float64)

    @property
    def widths(self) -> FloatArray:
        """Road width at each control point, shape ``(P,)``."""
        return np.array([p.width for p in self.control_points], dtype=np.float64)

    def to_track(self, **options: float) -> Track:
        """Derive the full track (see `Track.build`, which takes the same keyword options)."""
        return Track.build(self.points, self.widths, **options)


def read_track_file(path: str | Path) -> TrackFile:
    """Read and check a track file, upgrading it from an older format version if needed.

    Raises:
        TrackFileError: If the file can't be read or isn't a well-formed track file.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise TrackFileError(f"{path}: can't read the file ({error.strerror})") from error
    return parse_track_file(text, source=str(path))


def parse_track_file(text: str, source: str = "track file") -> TrackFile:
    """Check the text of a track file; ``source`` names it in error messages.

    Raises:
        TrackFileError: If the text isn't a well-formed track file.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise TrackFileError(
            f"{source}: not valid JSON (line {error.lineno}, column {error.colno}): {error.msg}"
        ) from error
    if not isinstance(data, dict):
        raise TrackFileError(f"{source}: expected a JSON object with the track's fields")
    return _validated(TrackFile, _upgrade(data, source), source)


def format_track_file(track_file: TrackFile) -> str:
    """The canonical text of a track file: two-space indentation, one control point per line."""
    data = track_file.model_dump(mode="json")
    fields = [f"  {json.dumps(key)}: {json.dumps(data[key], ensure_ascii=False)}" for key in
              ("schema_version", "name", "author", "description")]  # fmt: skip
    points = ",\n".join(f"    {json.dumps(point)}" for point in data["control_points"])
    fields.append(f'  "control_points": [\n{points}\n  ]')
    metadata = json.dumps(data["metadata"], indent=2, ensure_ascii=False).replace("\n", "\n  ")
    fields.append(f'  "metadata": {metadata}')
    return "{\n" + ",\n".join(fields) + "\n}\n"


def write_track_file(track_file: TrackFile, path: str | Path) -> None:
    """Save a track file in its canonical text form.

    The text goes to a temporary file first, which then replaces the target in one step, so a
    crash halfway through saving can't leave a half-written track behind.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(format_track_file(track_file), encoding="utf-8", newline="\n")
    temporary.replace(path)


def track_file_schema() -> str:
    """The JSON Schema of the current format, as published in the docs."""
    return json.dumps(TrackFile.model_json_schema(), indent=2) + "\n"


def _upgrade(data: dict[str, Any], source: str) -> dict[str, Any]:
    """Bring file data from its format version up to `CURRENT_VERSION`."""
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise TrackFileError(
            f"{source}: schema_version: must be a whole number of at least 1, got {version!r}"
        )
    if version > CURRENT_VERSION:
        raise TrackFileError(
            f"{source}: this track was saved in format version {version}, but this version of "
            f"MLRacecar can only read up to version {CURRENT_VERSION}. Update MLRacecar to open it."
        )
    while version < CURRENT_VERSION:
        if version not in MIGRATIONS:
            raise TrackFileError(f"{source}: no upgrade available from format version {version}")
        data = MIGRATIONS[version](data)
        version += 1
    return data


def _validated[Model: BaseModel](model: type[Model], data: dict[str, Any], source: str) -> Model:
    """Validate data into a model, turning pydantic's errors into one readable TrackFileError."""
    try:
        return model.model_validate(data)
    except ValidationError as error:
        problems = [
            f"  {'.'.join(str(part) for part in issue['loc']) or '(top level)'}: {issue['msg']}"
            for issue in error.errors()
        ]
        raise TrackFileError(f"{source}: not a valid track file:\n" + "\n".join(problems)) from None
