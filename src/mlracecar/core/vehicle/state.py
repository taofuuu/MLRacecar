"""The state of N cars: one array per quantity, each of shape ``(N,)`` (ADR-0005).

A car's position is its centre: the middle of the wheelbase, where the body is centred and the
centre of mass is assumed to be. Angles are counter-clockwise from +x, so positive steering,
yaw rate, and sideways speed all point to the car's left.
"""

from dataclasses import dataclass, fields
from typing import Self

import numpy as np
from numpy.typing import ArrayLike

from mlracecar.core.geometry import FloatArray, wrap_angle


@dataclass(frozen=True, eq=False)
class VehicleState:
    """Where N cars are and how they move.

    The arrays are never changed in place: each physics step makes a new state, so a state can
    be kept (for a replay, say) without copying.
    """

    x: FloatArray
    """Centre x in metres."""
    y: FloatArray
    """Centre y in metres."""
    yaw: FloatArray
    """Heading in radians, counter-clockwise from +x, in ``[-pi, pi)``."""
    vx: FloatArray
    """Speed forward, along the car, in m/s."""
    vy: FloatArray
    """Speed sideways, to the car's left, in m/s."""
    yaw_rate: FloatArray
    """How fast the heading turns, in rad/s (positive turns left)."""
    steer: FloatArray
    """Front-wheel angle in radians (positive points left)."""

    @classmethod
    def at_rest(cls, positions: ArrayLike, headings: ArrayLike) -> Self:
        """Cars standing still with their wheels straight, as on the starting grid.

        Args:
            positions: Centres, shape ``(N, 2)``.
            headings: Headings in radians, shape ``(N,)``.

        Raises:
            ValueError: If there aren't as many headings as positions.
        """
        xy = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        heading = wrap_angle(np.asarray(headings, dtype=np.float64).reshape(-1))
        if len(heading) != len(xy):
            raise ValueError(f"{len(xy)} positions but {len(heading)} headings")
        zeros = np.zeros(len(xy))
        return cls(xy[:, 0].copy(), xy[:, 1].copy(), heading, zeros, zeros, zeros, zeros)

    def __len__(self) -> int:
        """The number of cars."""
        return len(self.x)

    @property
    def position(self) -> FloatArray:
        """Centres, shape ``(N, 2)``."""
        return np.column_stack([self.x, self.y])

    @property
    def speed(self) -> FloatArray:
        """How fast each car's centre moves, in m/s."""
        result: FloatArray = np.hypot(self.vx, self.vy)
        return result

    def where(self, mask: ArrayLike, other: Self) -> Self:
        """These cars where ``mask`` is set, and ``other``'s cars elsewhere."""
        chosen = np.asarray(mask, dtype=bool)
        return type(self)(
            **{
                field.name: np.where(chosen, getattr(self, field.name), getattr(other, field.name))
                for field in fields(self)
            }
        )

    def select(self, cars: slice | ArrayLike) -> Self:
        """Some of the cars: ``cars`` picks them by index, by slice, or with a boolean mask."""
        index = cars if isinstance(cars, slice) else np.atleast_1d(np.asarray(cars))
        return type(self)(
            **{field.name: getattr(self, field.name)[index] for field in fields(self)}
        )
