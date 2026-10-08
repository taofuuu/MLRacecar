"""Replay files: a race recorded as its snapshots, to watch again (ADR-0015, `racecar replay`).

A replay keeps every snapshot of a race, one per driver decision (20 a second by default), with
the track it was driven on, every setting, and anything else worth knowing, such as who drove
and how it went. Snapshots are plain data, so playing a replay back shows exactly what happened
without running the simulation again, even after the code or the track file changes.

The file is NumPy's ``.npz``: a compressed zip of arrays. Each snapshot field is one array with
the snapshots along its first axis (``cars.x`` is ``(snapshots, cars)``), beside ``tick`` and
``time``; ``meta`` is a JSON text with the format and its version, the track file, the
settings, the race events, and the information. It holds no Python objects, so reading a replay
never runs code from it (``allow_pickle=False``). A one-car minute is about 110 kB.
"""

import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from mlracecar.core.race.events import LapCompleted, OffTrack, RaceEvent, WrongWay
from mlracecar.core.race.state import RaceState
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.io.track_file import TrackFile, TrackFileError, format_track_file, parse_track_file

FORMAT = "mlracecar-replay"
"""What ``meta`` says the file is."""

SCHEMA_VERSION = 1
"""The format's version. Reading refuses newer ones, naming the version."""

_EVENTS: dict[str, type[LapCompleted] | type[OffTrack] | type[WrongWay]] = {
    kind.__name__: kind for kind in (LapCompleted, OffTrack, WrongWay)
}


class ReplayError(ValueError):
    """A replay file that can't be read, or isn't a well-formed replay."""


@dataclass(frozen=True)
class Replay:
    """A recorded race."""

    snapshots: Sequence[Snapshot]
    """The race, a snapshot per decision from the start, each with the same cars."""
    track: TrackFile
    """The track it was driven on."""
    settings: Mapping[str, Any]
    """Every setting it ran with (`RacecarConfig`, as JSON)."""
    info: Mapping[str, Any]
    """Anything else worth knowing, such as who drove and how it went (JSON-ready)."""

    @property
    def duration(self) -> float:
        """Seconds of racing from the first snapshot to the last."""
        return self.snapshots[-1].time - self.snapshots[0].time


def write_replay(replay: Replay, path: str | Path) -> None:
    """Write a replay, making the folder if needed.

    Raises:
        ValueError: If there are no snapshots, or the cars change between them.
    """
    snapshots = replay.snapshots
    if not snapshots:
        raise ValueError("a replay needs at least one snapshot")
    if len({len(snapshot.cars) for snapshot in snapshots}) > 1:
        raise ValueError("every snapshot of a replay must have the same cars")
    columns: dict[str, Any] = {  # Any: NumPy's hints take ** for its own options
        "tick": np.array([snapshot.tick for snapshot in snapshots], dtype=np.int64),
        "time": np.array([snapshot.time for snapshot in snapshots]),
    }
    for part in ("cars", "race"):
        for field in fields(getattr(snapshots[0], part)):
            columns[f"{part}.{field.name}"] = np.stack(
                [getattr(getattr(snapshot, part), field.name) for snapshot in snapshots]
            )
    meta = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "track": json.loads(format_track_file(replay.track)),
        "settings": replay.settings,
        "info": replay.info,
        "events": [
            {"step": step, "kind": type(event).__name__} | _plain(asdict(event))
            for step, snapshot in enumerate(snapshots)
            for event in snapshot.events
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as file:  # a file, so NumPy doesn't add ".npz" to the name
        np.savez_compressed(file, meta=np.array(json.dumps(meta, allow_nan=False)), **columns)


def read_replay(path: str | Path) -> Replay:
    """Read a replay. Its snapshots' arrays are read-only, as the world's are.

    Raises:
        ReplayError: If the file can't be read, isn't a replay, or was made by a newer version.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except OSError as error:
        raise ReplayError(f"{path}: can't read the replay ({error.strerror or error})") from error
    except (ValueError, zipfile.BadZipFile, EOFError) as error:
        raise ReplayError(f"{path}: not a replay file ({error})") from error
    meta = _meta(arrays, path)
    names = ["tick", "time"]
    names += [f"cars.{field.name}" for field in fields(VehicleState)]
    names += [f"race.{field.name}" for field in fields(RaceState)]
    missing = [name for name in names if name not in arrays]
    if missing:
        raise ReplayError(f"{path}: not a replay file (no {', '.join(missing)})")
    if len({len(arrays[name]) for name in names}) > 1:
        raise ReplayError(f"{path}: not a replay file (its arrays differ in length)")
    for name in names:
        arrays[name].flags.writeable = False
    try:
        track = parse_track_file(json.dumps(meta["track"]), source=f"{path}: its track")
    except TrackFileError as error:
        raise ReplayError(str(error)) from error
    events = _events(meta["events"], len(arrays["tick"]), path)
    snapshots = tuple(
        Snapshot(
            int(arrays["tick"][step]),
            float(arrays["time"][step]),
            VehicleState(**{f.name: arrays[f"cars.{f.name}"][step] for f in fields(VehicleState)}),
            RaceState(**{f.name: arrays[f"race.{f.name}"][step] for f in fields(RaceState)}),
            events[step],
        )
        for step in range(len(arrays["tick"]))
    )
    if not snapshots:
        raise ReplayError(f"{path}: not a replay file (no snapshots)")
    return Replay(snapshots, track, meta["settings"], meta["info"])


def _meta(arrays: dict[str, np.ndarray], path: Path) -> dict[str, Any]:
    """The replay's ``meta``, checked."""
    if "meta" not in arrays:
        raise ReplayError(f"{path}: not a replay file (no meta)")
    try:
        meta = json.loads(str(arrays.pop("meta")))
    except json.JSONDecodeError as error:
        raise ReplayError(f"{path}: not a replay file (its meta isn't JSON)") from error
    if not isinstance(meta, dict) or meta.get("format") != FORMAT:
        raise ReplayError(f"{path}: not a replay file")
    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ReplayError(
            f"{path}: replay format version {version}; this version of MLRacecar reads version "
            f"{SCHEMA_VERSION} (a newer MLRacecar may read it)"
        )
    for key in ("track", "settings", "info", "events"):
        if key not in meta:
            raise ReplayError(f"{path}: not a replay file (its meta has no {key})")
    return meta


def _events(entries: list[dict[str, Any]], steps: int, path: Path) -> list[tuple[RaceEvent, ...]]:
    """The race events, grouped by the snapshot they belong to."""
    grouped: list[list[RaceEvent]] = [[] for _ in range(steps)]
    for entry in entries:
        values = dict(entry)
        step, kind = values.pop("step"), _EVENTS.get(values.pop("kind"))
        if kind is None or not 0 <= step < steps:
            raise ReplayError(f"{path}: not a replay file (an event it can't place)")
        if "sectors" in values:
            values["sectors"] = tuple(
                math.nan if time is None else time for time in values["sectors"]
            )
        grouped[step].append(kind(**values))
    return [tuple(events) for events in grouped]


def _plain(value: Any) -> Any:
    """Event data as standard JSON: NaN (a sector whose start was missed) becomes null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value
