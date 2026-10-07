"""Golden trajectories: recorded runs of the simulation that later runs must match.

Each `Scenario` drives cars round a sample track with scripted actions, through the whole
simulation (car physics and race rules), and records every car's state and race after each
driver decision. The recordings live in ``tests/regression/golden/`` and are compared with fresh
runs by ``tests/regression/test_golden.py``. If the physics change by accident, the comparison
says which quantities changed, when, and by how much.

When a change is intended, record them again and say why in the pull request:

    uv run python scripts/update_golden.py
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from drivers import CenterlineDriver
from mlracecar.config.models import RacecarConfig
from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.race.events import LapCompleted
from mlracecar.core.race.rules import OffTrackPolicy, RaceSettings
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.kinematic import KinematicBicycle
from mlracecar.core.vehicle.state import VehicleState
from mlracecar.core.world import World
from mlracecar.io.track_file import read_track_file

GOLDEN_DIR = Path(__file__).parent / "regression" / "golden"
TRACKS = Path(__file__).parents[1] / "tracks"

ABSOLUTE = 1e-6
"""How far a recorded number may drift, plus `RELATIVE` of its size. Windows and Linux compute
sin, cos, and friends very slightly differently, so runs agree to about 1e-12, not exactly; a
real change to the physics moves things by far more than this."""
RELATIVE = 1e-6

ANGLES = frozenset({"yaw", "heading_error"})
"""Quantities compared the short way round, so -pi and +pi count as the same."""

type Actions = Callable[[Snapshot, int], FloatArray]
"""Each car's ``[steer, pedal]`` for a snapshot and the decision's number."""
type Recording = dict[str, NDArray[np.generic]]


@dataclass(frozen=True)
class Scenario:
    """A scripted run to record."""

    name: str
    track: str
    """A sample track's file name in ``tracks/``."""
    cars: int
    seconds: float
    actions: Callable[[Track], Actions]
    """Makes the actions for the track (a scripted driver needs to know it)."""
    off_track: OffTrackPolicy = OffTrackPolicy.SLOWDOWN
    config: RacecarConfig = field(default_factory=RacecarConfig)


def _straight_line(track: Track) -> Actions:
    """Full throttle for 4 s, coast for 1 s, then full braking: the engine, drag, and brakes."""

    def act(snapshot: Snapshot, decision: int) -> FloatArray:
        pedal = 1.0 if decision < 80 else 0.0 if decision < 100 else -1.0
        return np.array([[0.0, pedal]])

    return act


def _slalom(track: Track) -> Actions:
    """Weaving at half throttle, harder and harder: the steering rate and the grip limit."""

    def act(snapshot: Snapshot, decision: int) -> FloatArray:
        time = decision * 0.05
        return np.array([[min(1.0, 0.1 + time / 15) * math.sin(math.pi * time), 0.5]])

    return act


def _scripted_lap(track: Track) -> Actions:
    """The scripted driver lapping the track: progress, checkpoints, laps, and times."""
    driver = CenterlineDriver(track, RacecarConfig().vehicle.to_params(), 60.0)
    return lambda snapshot, decision: driver.act(snapshot)


def _mixed_batch(track: Track) -> Actions:
    """Four cars from the grid: careful, off the road, braking hard, and turning round.

    Car 1 steers off the road at speed (the grass slows it); car 3 turns round at walking pace
    and drives back the wrong way.
    """
    driver = CenterlineDriver(track, RacecarConfig().vehicle.to_params(), 30.0)

    def act(snapshot: Snapshot, decision: int) -> FloatArray:
        careful = driver.act(snapshot)
        time = decision * 0.05
        return np.array([
            careful[0],
            [0.0, 1.0] if time < 2.0 else [-1.0, 1.0],
            careful[2] if time < 3.0 else [0.0, -1.0],
            [1.0, 0.25] if time < 5.0 else [0.0, 1.0],
        ])  # fmt: skip

    return act


SCENARIOS = (
    Scenario(
        "straight-line", "gp-circuit.json", 1, 7.0, _straight_line, off_track=OffTrackPolicy.NONE
    ),
    Scenario("slalom", "oval.json", 1, 15.0, _slalom, off_track=OffTrackPolicy.NONE),
    Scenario("scripted-lap", "technical.json", 1, 75.0, _scripted_lap),
    Scenario("mixed-batch", "technical.json", 4, 10.0, _mixed_batch),
)


def record(scenario: Scenario) -> Recording:
    """Run a scenario and record every car's state and race after each decision."""
    config = scenario.config
    track = read_track_file(TRACKS / scenario.track).to_track()
    world = World(
        track,
        KinematicBicycle(config.vehicle.to_params()),
        config.simulation.to_timing(),
        scenario.cars,
        np.random.default_rng(0),
        settings=RaceSettings(
            off_track=scenario.off_track, grass_slowdown=config.race.grass_slowdown
        ),
    )
    actions = scenario.actions(track)
    snapshots = []
    for decision in range(round(scenario.seconds / world.timing.decision_dt)):
        snapshots.append(world.step(actions(world.snapshot, decision)))
    return _arrays(snapshots)


def _arrays(snapshots: list[Snapshot]) -> Recording:
    columns: Recording = {"time": np.array([snapshot.time for snapshot in snapshots])}
    for name in (item.name for item in fields(VehicleState)):
        columns[name] = np.stack([getattr(snapshot.cars, name) for snapshot in snapshots])
    for name in ("distance", "offset", "heading_error", "checkpoint", "laps", "off_track",
                 "wrong_way", "out"):  # fmt: skip
        columns[name] = np.stack([getattr(snapshot.race, name) for snapshot in snapshots])
    laps = [event for snapshot in snapshots for event in snapshot.events
            if isinstance(event, LapCompleted)]  # fmt: skip
    columns["lap_car"] = np.array([lap.car for lap in laps], dtype=np.intp)
    columns["lap_time"] = np.array([lap.time for lap in laps], dtype=np.float64)
    columns["lap_valid"] = np.array([lap.valid for lap in laps], dtype=bool)
    return columns


def golden_path(scenario: Scenario) -> Path:
    return GOLDEN_DIR / f"{scenario.name}.npz"


def write(scenario: Scenario) -> Path:
    """Record a scenario and save it as its golden file."""
    path = golden_path(scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **record(scenario))  # type: ignore[arg-type]
    return path


def read(scenario: Scenario) -> Recording:
    with np.load(golden_path(scenario)) as saved:
        return {name: saved[name] for name in saved.files}


def differences(scenario: Scenario, golden: Recording, now: Recording) -> str | None:
    """A readable summary of how ``now`` differs from ``golden``, or ``None`` if it matches."""
    problems = []
    for name in sorted(golden.keys() | now.keys()):
        if name not in golden or name not in now:
            problems.append(
                (-1, f"  {name}: only in the {'new run' if name in now else 'golden file'}")
            )
            continue
        expected, actual = golden[name], now[name]
        if expected.shape != actual.shape:
            problems.append((-1, f"  {name}: shape {actual.shape}, golden {expected.shape}"))
            continue
        gap = actual.astype(np.float64) - expected.astype(np.float64)
        if expected.dtype.kind == "f":
            if name in ANGLES:
                gap = wrap_angle(gap)
            wrong = np.abs(gap) > ABSOLUTE + RELATIVE * np.abs(expected.astype(np.float64))
        else:
            wrong = actual != expected
        if not wrong.any():
            continue
        first = np.unravel_index(int(np.argmax(wrong)), wrong.shape)
        step = int(first[0])
        when = (
            f"step {step} (t = {golden['time'][step]:.2f} s)"
            if expected.ndim == 2
            else f"entry {step}"
        )
        car = f", car {int(first[1])}" if expected.ndim == 2 else ""
        problems.append((
            step,
            f"  {name}: first differs at {when}{car}: golden {_show(expected[first])}, now "
            f"{_show(actual[first])} ({abs(float(gap[first])):.3g} apart); at most "
            f"{float(np.max(np.abs(gap))):.3g} apart",
        ))  # fmt: skip
    if not problems:
        return None
    lines = [f"The '{scenario.name}' run no longer matches its golden trajectory:"]
    lines += [line for _, line in sorted(problems)]
    lines += [
        "If the physics or race rules changed on purpose, record the golden files again with",
        "    uv run python scripts/update_golden.py",
        "and say why in the pull request.",
    ]
    return "\n".join(lines)


def _show(value: np.generic) -> str:
    """Enough digits to see two nearly equal numbers apart."""
    return f"{value:.9g}" if isinstance(value, np.floating) else str(value)
