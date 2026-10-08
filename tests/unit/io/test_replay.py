"""Tests for mlracecar.io.replay: recorded races, written and read back exactly."""

import io
import json
import math
import zipfile
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from mlracecar.config.models import EpisodeConfig, RacecarConfig
from mlracecar.core.race.events import LapCompleted, OffTrack, WrongWay
from mlracecar.core.snapshot import Snapshot
from mlracecar.env.racing import RacingEnv
from mlracecar.io.replay import Replay, ReplayError, read_replay, write_replay
from mlracecar.io.track_file import TrackFile

ANGLES = np.linspace(0, 2 * np.pi, 48, endpoint=False)
CIRCLE = TrackFile.from_arrays(
    "Circle", 60 * np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), [12.0] * 48
)
CONFIG = RacecarConfig(episode=EpisodeConfig(time_limit=60.0))


def circling(observations: NDArray[np.float32]) -> NDArray[np.float32]:
    """About 20 m/s round the circle, steering by the offset input."""
    speed, offset = observations[15] * 100, observations[18]
    steer = np.clip(0.33 - 0.5 * offset, -1, 1)
    return np.array([steer, np.clip((20 - speed) * 0.3, -1, 1)], dtype=np.float32)


def race(seconds: float) -> list[Snapshot]:
    """A car's snapshots, from the grid, circling for ``seconds``."""
    env = RacingEnv(CIRCLE.to_track(), CONFIG)
    observation, _ = env.reset(seed=0)
    assert env.world is not None
    snapshots = [env.world.snapshot]
    for _ in range(round(seconds / 0.05)):
        observation, *_ = env.step(circling(observation))
        snapshots.append(env.world.snapshot)
    return snapshots


def with_events(snapshots: list[Snapshot]) -> list[Snapshot]:
    """The snapshots, with every kind of event added to two of them (one with a missed sector)."""
    lap = LapCompleted(0, 31.5, (10.0, math.nan, 11.5), False, 3.0)
    snapshots[3] = replace(snapshots[3], events=(lap, OffTrack(0, 120.5, 3.0)))
    snapshots[5] = replace(snapshots[5], events=(WrongWay(0, 118.0, 5.1),))
    return snapshots


def replay_of(snapshots: list[Snapshot]) -> Replay:
    return Replay(
        snapshots, CIRCLE, CONFIG.model_dump(mode="json"), {"driver": "circling", "laps": 3}
    )


def assert_same_race(first: list[Snapshot], second: list[Snapshot]) -> None:
    """Snapshot by snapshot, field by field, bit for bit; events too (NaN equal to NaN)."""
    assert len(first) == len(second)
    for one, other in zip(first, second, strict=True):
        assert (one.tick, one.time) == (other.tick, other.time)
        for part in ("cars", "race"):
            for field in fields(getattr(one, part)):
                np.testing.assert_array_equal(
                    getattr(getattr(one, part), field.name),
                    getattr(getattr(other, part), field.name),
                    err_msg=f"{part}.{field.name}",
                )
        assert repr(one.events) == repr(other.events)


def test_a_replay_reads_back_exactly_as_it_was_written(tmp_path: Path) -> None:
    snapshots = with_events(race(2.0))
    replay = replay_of(snapshots)

    write_replay(replay, tmp_path / "circle.npz")
    back = read_replay(tmp_path / "circle.npz")

    assert_same_race(list(back.snapshots), snapshots)
    assert back.track == CIRCLE
    assert back.settings == CONFIG.model_dump(mode="json")
    assert back.info == {"driver": "circling", "laps": 3}
    assert back.duration == pytest.approx(2.0)


def test_a_replay_read_back_cant_be_changed(tmp_path: Path) -> None:
    write_replay(replay_of(race(0.5)), tmp_path / "circle.npz")

    snapshot = read_replay(tmp_path / "circle.npz").snapshots[-1]

    with pytest.raises(ValueError, match="read-only"):
        snapshot.cars.x[0] = 0.0


def test_a_replay_is_plain_arrays_and_json(tmp_path: Path) -> None:
    write_replay(replay_of(with_events(race(0.5))), tmp_path / "replay")  # no ".npz" added

    with np.load(tmp_path / "replay", allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        assert data["cars.x"].shape == (11, 1)
        assert data["race.splits"].shape == (11, 1, 2)
    assert (meta["format"], meta["schema_version"]) == ("mlracecar-replay", 1)
    assert meta["track"]["name"] == "Circle"
    assert meta["events"][0] == {
        "step": 3,
        "kind": "LapCompleted",
        "car": 0,
        "time": 31.5,
        "sectors": [10.0, None, 11.5],
        "valid": False,
        "at": 3.0,
    }


def test_a_minute_of_racing_takes_far_less_than_a_megabyte(tmp_path: Path) -> None:
    snapshots = race(60.0)
    assert snapshots[-1].race.laps[0] >= 2

    write_replay(replay_of(snapshots), tmp_path / "minute.npz")

    assert (tmp_path / "minute.npz").stat().st_size < 1_000_000


def test_a_replay_needs_snapshots_of_the_same_cars(tmp_path: Path) -> None:
    snapshots = race(0.1)
    with pytest.raises(ValueError, match="at least one snapshot"):
        write_replay(replay_of([]), tmp_path / "empty.npz")
    two_cars = replace(snapshots[1], cars=snapshots[1].cars.select([0, 0]))
    with pytest.raises(ValueError, match="the same cars"):
        write_replay(replay_of([snapshots[0], two_cars]), tmp_path / "mixed.npz")


# --------------------------------------------------------------------------- #
# Files that aren't replays
# --------------------------------------------------------------------------- #


def npz(path: Path, **arrays: Any) -> Path:
    with path.open("wb") as file:
        np.savez(file, **arrays)
    return path


def replay_arrays(tmp_path: Path) -> dict[str, Any]:
    write_replay(replay_of(race(0.1)), tmp_path / "good.npz")
    with np.load(tmp_path / "good.npz") as data:
        return {key: data[key] for key in data.files}


def meta_with(arrays: dict[str, Any], **changes: Any) -> dict[str, Any]:
    meta = json.loads(str(arrays["meta"])) | changes
    return arrays | {"meta": np.array(json.dumps(meta))}


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match=r"missing\.npz: can't read the replay"):
        read_replay(tmp_path / "missing.npz")


def test_a_file_that_isnt_a_zip_of_arrays_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "notes.npz"
    path.write_text("not a replay", encoding="utf-8")

    with pytest.raises(ReplayError, match=r"notes\.npz: not a replay file"):
        read_replay(path)


def test_a_replay_holding_python_objects_is_refused(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"code": "anything"}], dtype=object), allow_pickle=True)
    path = tmp_path / "pickled.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("meta.npy", buffer.getvalue())

    with pytest.raises(ReplayError, match="not a replay file"):
        read_replay(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"meta": None}, "no meta"),
        ({"meta": np.array("{not json")}, "its meta isn't JSON"),
        ({"meta": np.array('{"format": "something else"}')}, r"other.npz: not a replay file$"),
        ({"cars.x": None, "race.laps": None}, r"no cars\.x, race\.laps"),
        ({"time": np.zeros(7)}, "its arrays differ in length"),
    ],
)
def test_arrays_that_arent_a_replay_are_refused(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    arrays = replay_arrays(tmp_path) | change
    path = npz(tmp_path / "other.npz", **{k: v for k, v in arrays.items() if v is not None})

    with pytest.raises(ReplayError, match=message):
        read_replay(path)


def test_a_replay_without_snapshots_is_refused(tmp_path: Path) -> None:
    arrays = replay_arrays(tmp_path)
    empty = {key: value if key == "meta" else value[:0] for key, value in arrays.items()}

    with pytest.raises(ReplayError, match="no snapshots"):
        read_replay(npz(tmp_path / "empty.npz", **empty))


def test_a_replay_from_a_newer_version_says_so(tmp_path: Path) -> None:
    path = npz(tmp_path / "new.npz", **meta_with(replay_arrays(tmp_path), schema_version=2))

    with pytest.raises(ReplayError, match="replay format version 2; this version of MLRacecar"):
        read_replay(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"info": None}, "its meta has no info"),
        ({"track": {"name": "No points"}}, "its track"),
        ({"events": [{"step": 0, "kind": "Crash", "car": 0}]}, "an event it can't place"),
        ({"events": [{"step": 99, "kind": "OffTrack", "car": 0}]}, "an event it can't place"),
    ],
)
def test_a_replay_with_bad_meta_is_refused(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    arrays = replay_arrays(tmp_path)
    meta = {k: v for k, v in (json.loads(str(arrays["meta"])) | changes).items() if v is not None}
    path = npz(tmp_path / "bad.npz", **(arrays | {"meta": np.array(json.dumps(meta))}))

    with pytest.raises(ReplayError, match=message):
        read_replay(path)
