"""Race rules: progress along the lap, checkpoints, laps, and lap and sector times.

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
"""

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

from mlracecar.core.geometry import BoolArray, FloatArray, IntArray, cross, wrap_angle
from mlracecar.core.race.events import LapCompleted, RaceEvent
from mlracecar.core.race.progress import RoadLocator
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.state import VehicleState

SECTORS = 3
"""Sectors per lap, as in real racing."""


@dataclass(frozen=True, eq=False)
class RaceState:
    """Each car's race, one array per quantity. Arrays have shape ``(N,)`` unless noted."""

    segment: IntArray
    """The centerline segment each car was found on, where the next search starts."""
    arc_length: FloatArray
    """Distance along the lap from the start/finish line, in ``[0, track length)`` metres."""
    distance: FloatArray
    """Metres driven along the track since the car started; driving backwards subtracts."""
    offset: FloatArray
    """Distance from the centerline in metres, positive to the left."""
    heading_error: FloatArray
    """The car's heading minus the road's, in ``[-pi, pi)``: 0 is straight along the road."""
    checkpoint: IntArray
    """The last checkpoint crossed this lap; -1 until the lap starts at checkpoint 0."""
    clean: BoolArray
    """Whether the car has crossed every checkpoint in order so far this lap."""
    lap_start: FloatArray
    """The world's time when the current lap started; NaN until it starts."""
    splits: FloatArray
    """Time into the current lap when each sector after the first began, shape
    ``(N, SECTORS - 1)``; NaN until then."""
    laps: IntArray
    """Valid laps completed."""
    last_lap: FloatArray
    """The latest valid lap time in seconds; NaN until there is one."""
    best_lap: FloatArray
    """The best valid lap time in seconds; NaN until there is one."""


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
    """

    def __init__(self, track: Track, reach: float) -> None:
        self.track = track
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
        )

    def update(
        self,
        race: RaceState,
        before: VehicleState,
        after: VehicleState,
        start: float,
        end: float,
    ) -> tuple[RaceState, tuple[RaceEvent, ...]]:
        """The race after the cars moved from ``before`` (at time ``start``) to ``after``."""
        where = self._locator.locate(after.position, near=race.segment)
        half_lap = self._locator.length / 2
        moved = np.mod(where.arc_length - race.arc_length + half_lap, 2 * half_lap) - half_lap
        moved_on = replace(
            race,
            segment=where.segment,
            arc_length=where.arc_length,
            distance=race.distance + moved,
            offset=where.offset,
            heading_error=wrap_angle(after.yaw - where.heading),
        )
        crossings = self._crossings(race.arc_length, where.arc_length, before, after, start, end)
        if not crossings:
            return moved_on, ()
        return self._crossed(moved_on, crossings)

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
                    valid = bool(clean[car]) and checkpoint[car] == self._count - 1
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
