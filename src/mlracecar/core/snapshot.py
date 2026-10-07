"""A picture of the race at one moment, as plain data (architecture section 4.4).

`mlracecar.core.world.World` hands one out after every step. Everything that only looks at
the race, such as the renderer, a replay, or an environment's observations, works from
snapshots alone.
"""

from dataclasses import dataclass

from mlracecar.core.race.events import RaceEvent
from mlracecar.core.race.state import RaceState
from mlracecar.core.vehicle.state import VehicleState


@dataclass(frozen=True)
class Snapshot:
    """The world at one moment. Its arrays are read-only, so it can be kept and shared."""

    tick: int
    """Physics steps since the world began."""
    time: float
    """Simulated seconds since the world began."""
    cars: VehicleState
    """Every car's position and motion."""
    race: RaceState
    """Every car's race: where it is along the lap, its laps, and its times."""
    events: tuple[RaceEvent, ...]
    """What happened since the previous snapshot, in order."""
