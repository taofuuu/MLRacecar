"""Race rules: progress along the lap, checkpoints, laps, times, and leaving the road.

Checkpoints are lines across the road (`Track.checkpoints`); checkpoint 0 is the start/finish
line. A lap starts when a car crosses the start/finish line forwards and ends when it next does.
It counts, as a *valid* lap, only if the car crossed every checkpoint in between in order:

- Crossing a checkpoint backwards undoes crossing it forwards, so reversing back and forth over
  the line (or any checkpoint) gains nothing.
- The lines end at the road's edges, so cutting a corner past a checkpoint, or going round it
  off the road, misses it and makes the lap invalid.

A car that starts behind the line (on the grid) or anywhere else along the lap starts its first
lap when it reaches the line. Each lap is split into `SECTORS` sectors, which begin at evenly
spread checkpoints. Times are interpolated between updates, so they are accurate to far better
than one update.

A car is *off track* while its centre is off the road, and what happens then is the
`OffTrackPolicy`: nothing, the grass slows it down, it is put back on the road, or its run is
over. A car drives the *wrong way* while it moves backwards along the track faster than
`WRONG_WAY_SPEED`. Each time either starts, an event says so.
"""

import itertools
import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from mlracecar.core.geometry import FloatArray, cross, wrap_angle
from mlracecar.core.race.events import LapCompleted, OffTrack, RaceEvent, WrongWay
from mlracecar.core.race.progress import RoadLocator
from mlracecar.core.race.state import SECTORS, RaceState
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.state import VehicleState

WRONG_WAY_SPEED = 1.0
"""m/s backwards along the track from which a car counts as driving the wrong way."""


class OffTrackPolicy(StrEnum):
    """What happens to a car whose centre leaves the road."""

    NONE = "none"
    """Nothing: the car carries on (the event still says it happened)."""
    SLOWDOWN = "slowdown"
    """The grass slows it down for as long as it is off the road."""
    RESET = "reset"
    """It is put back in the middle of the road where it left it, at rest."""
    TERMINATE = "terminate"
    """Its run is over: it stops where it is until it is reset."""


@dataclass(frozen=True)
class RaceSettings:
    """The rules' settings. `mlracecar.config.models.RaceConfig` builds them from settings files;
    the defaults here are the same."""

    off_track: OffTrackPolicy = OffTrackPolicy.SLOWDOWN
    """What happens to a car whose centre leaves the road."""
    grass_slowdown: float = 6.0
    """How hard the grass slows a car down, in m/s² (with `OffTrackPolicy.SLOWDOWN`)."""


@dataclass(frozen=True)
class _Crossing:
    car: int
    checkpoint: int
    forward: bool
    at: float


class RaceRules:
    """Applies the race rules on a track.

    Args:
        track: The track.
        reach: The farthest a car can move between two updates, in metres.
        settings: What happens when a car leaves the road.
    """

    def __init__(self, track: Track, reach: float, settings: RaceSettings | None = None) -> None:
        self.track = track
        self.settings = settings or RaceSettings()
        self._locator = RoadLocator(track, reach)
        checkpoints = track.checkpoints
        self._left = checkpoints.left
        self._line = checkpoints.right - checkpoints.left
        self._count = len(checkpoints.arc_length)
        self._sector_starts = {
            round(sector * self._count / SECTORS): sector - 1 for sector in range(1, SECTORS)
        }

    def start(self, cars: VehicleState) -> RaceState:
        """A new race for cars that just started, wherever they are: no lap begun yet."""
        where = self._locator.locate(cars.position)
        count = len(cars)
        return RaceState(
            segment=where.segment,
            arc_length=where.arc_length,
            distance=np.zeros(count),
            offset=where.offset,
            heading_error=wrap_angle(cars.yaw - where.heading),
            checkpoint=np.full(count, -1),
            clean=np.ones(count, dtype=bool),
            lap_start=np.full(count, np.nan),
            splits=np.full((count, SECTORS - 1), np.nan),
            laps=np.zeros(count, dtype=np.intp),
            last_lap=np.full(count, np.nan),
            best_lap=np.full(count, np.nan),
            off_track=np.abs(where.offset) > where.width / 2,
            wrong_way=np.zeros(count, dtype=bool),
            out=np.zeros(count, dtype=bool),
        )

    def update(
        self,
        race: RaceState,
        before: VehicleState,
        after: VehicleState,
        start: float,
        end: float,
    ) -> tuple[RaceState, tuple[RaceEvent, ...]]:
        """The race after the cars moved from ``before`` (at time ``start``) to ``after``.

        Follow it with `enforce`, which applies the off-track policy.
        """
        where = self._locator.locate(after.position, near=race.segment)
        half_lap = self._locator.length / 2
        moved = np.mod(where.arc_length - race.arc_length + half_lap, 2 * half_lap) - half_lap
        off_track = np.abs(where.offset) > where.width / 2
        wrong_way = moved < -WRONG_WAY_SPEED * (end - start)
        moved_on = replace(
            race,
            segment=where.segment,
            arc_length=where.arc_length,
            distance=race.distance + moved,
            offset=where.offset,
            heading_error=wrap_angle(after.yaw - where.heading),
            off_track=off_track,
            wrong_way=wrong_way,
            out=race.out | (off_track & (self.settings.off_track is OffTrackPolicy.TERMINATE)),
        )
        events: list[RaceEvent] = [
            *(OffTrack(int(car), float(where.arc_length[car]), end)
              for car in np.flatnonzero(off_track & ~race.off_track)),
            *(WrongWay(int(car), float(where.arc_length[car]), end)
              for car in np.flatnonzero(wrong_way & ~race.wrong_way)),
        ]  # fmt: skip
        crossings = self._crossings(race.arc_length, where.arc_length, before, after, start, end)
        if crossings:
            moved_on, laps = self._crossed(moved_on, crossings)
            events = [*laps, *events]  # laps happen during the update, the others at its end
        return moved_on, tuple(events)

    def enforce(
        self, cars: VehicleState, race: RaceState, dt: float
    ) -> tuple[VehicleState, RaceState]:
        """Apply the off-track policy to cars that are off the road, after an update of ``dt``
        seconds."""
        policy = self.settings.off_track
        if policy is OffTrackPolicy.SLOWDOWN:
            speed = cars.speed
            slower = np.maximum(speed - self.settings.grass_slowdown * dt, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(race.off_track & (speed > 0), slower / speed, 1.0)
            return _scaled(cars, scale), race
        if policy is OffTrackPolicy.RESET and race.off_track.any():
            pose = self.track.pose_at(race.arc_length)
            placed = VehicleState.at_rest(pose.position, pose.heading)
            back = race.off_track
            return placed.where(back, cars), replace(
                race,
                offset=np.where(back, 0.0, race.offset),
                heading_error=np.where(back, 0.0, race.heading_error),
                off_track=np.zeros_like(back),
            )
        if policy is OffTrackPolicy.TERMINATE:
            return _scaled(cars, np.where(race.out, 0.0, 1.0)), race
        return cars, race

    def _crossings(
        self,
        arc_before: FloatArray,
        arc_after: FloatArray,
        before: VehicleState,
        after: VehicleState,
        start: float,
        end: float,
    ) -> list[_Crossing]:
        """Checkpoint lines the cars crossed, in the order they crossed them.

        Each car is checked against the checkpoints nearest to where it was and where it is,
        which finds every crossing while cars move less than half the checkpoint spacing
        between updates.
        """
        spacing = self._locator.length / self._count
        found = []
        nearest_before = np.rint(arc_before / spacing).astype(np.intp) % self._count
        nearest_after = np.rint(arc_after / spacing).astype(np.intp) % self._count
        for candidate, check in (
            (nearest_before, True),
            (nearest_after, nearest_after != nearest_before),
        ):
            left, line = self._left[candidate], self._line[candidate]
            side_before = cross(line, before.position - left)
            side_after = cross(line, after.position - left)  # positive: past the line
            # Cars that stayed on one side divide by zero here; they never count as crossing.
            with np.errstate(divide="ignore", invalid="ignore"):
                fraction = side_before / (side_before - side_after)
                point = before.position + fraction[:, None] * (after.position - before.position)
                along = np.einsum("ni,ni->n", point - left, line) / np.einsum(
                    "ni,ni->n", line, line
                )
            switched = (side_before > 0) != (side_after > 0)
            crossed = check & switched & (along >= 0) & (along <= 1)
            for car in np.flatnonzero(crossed):
                at = start + float(fraction[car]) * (end - start)
                found.append(
                    _Crossing(int(car), int(candidate[car]), bool(side_after[car] > 0), at)
                )
        return sorted(found, key=lambda crossing: (crossing.at, crossing.car))

    def _crossed(
        self, race: RaceState, crossings: list[_Crossing]
    ) -> tuple[RaceState, tuple[RaceEvent, ...]]:
        """Apply checkpoint crossings to the cars' laps."""
        checkpoint, clean = race.checkpoint.copy(), race.clean.copy()
        lap_start, splits = race.lap_start.copy(), race.splits.copy()
        laps, last_lap, best_lap = race.laps.copy(), race.last_lap.copy(), race.best_lap.copy()
        events: list[RaceEvent] = []
        for crossing in crossings:
            car, line = crossing.car, crossing.checkpoint
            sector = self._sector_starts.get(line)
            if not crossing.forward:
                if line == checkpoint[car]:  # backing over the last line crossed undoes it
                    checkpoint[car] = line - 1
                    if line == 0:
                        lap_start[car] = np.nan
                    if sector is not None:
                        splits[car, sector] = np.nan
            elif line == 0:
                if checkpoint[car] > 0:  # a lap was under way: it ends here
                    time = crossing.at - float(lap_start[car])
                    valid = bool(clean[car] and checkpoint[car] == self._count - 1)
                    edges = [0.0, *splits[car].tolist(), time]
                    sectors = tuple(after - before for before, after in itertools.pairwise(edges))
                    events.append(LapCompleted(car, time, sectors, valid, crossing.at))
                    if valid:
                        laps[car] += 1
                        last_lap[car] = time
                        best_lap[car] = (
                            time if math.isnan(best_lap[car]) else min(best_lap[car], time)
                        )
                checkpoint[car], clean[car], lap_start[car] = 0, True, crossing.at
                splits[car] = np.nan
            elif checkpoint[car] >= 0 and line > checkpoint[car]:
                if line != checkpoint[car] + 1:
                    clean[car] = False  # a checkpoint was missed
                checkpoint[car] = line
                if sector is not None:
                    splits[car, sector] = crossing.at - lap_start[car]
        crossed = replace(
            race,
            checkpoint=checkpoint,
            clean=clean,
            lap_start=lap_start,
            splits=splits,
            laps=laps,
            last_lap=last_lap,
            best_lap=best_lap,
        )
        return crossed, tuple(events)


def _scaled(cars: VehicleState, scale: FloatArray) -> VehicleState:
    """The cars with their motion scaled: 0 stops a car, 1 leaves it as it is."""
    return replace(cars, vx=cars.vx * scale, vy=cars.vy * scale, yaw_rate=cars.yaw_rate * scale)
