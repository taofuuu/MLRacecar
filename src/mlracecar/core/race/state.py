"""Each car's race as plain data: where it is along the lap, its laps, and its times.

`mlracecar.core.race.rules` keeps it up to date; renderers and environments only read it.
"""

from dataclasses import dataclass, fields
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import BoolArray, FloatArray, IntArray

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
    off_track: BoolArray
    """Whether the car's centre is off the road."""
    wrong_way: BoolArray
    """Whether the car is driving backwards along the track."""
    out: BoolArray
    """Whether the car's run is over (it left the road under `OffTrackPolicy.TERMINATE`); it
    stays where it is, at rest, until it is reset."""

    def select(self, cars: slice | ArrayLike) -> Self:
        """Some of the cars' races: ``cars`` picks them by index, by slice, or with a mask."""
        index = cars if isinstance(cars, slice) else np.atleast_1d(np.asarray(cars))
        return type(self)(
            **{field.name: getattr(self, field.name)[index] for field in fields(self)}
        )
