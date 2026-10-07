"""Hypothesis strategies shared by several test modules."""

import numpy as np
from hypothesis import strategies as st

from mlracecar.core.geometry import FloatArray, wrap_angle
from mlracecar.core.vehicle.state import VehicleState


@st.composite
def track_like_points(draw: st.DrawFn, *, clockwise: bool | None = False) -> FloatArray:
    """8-16 points on an ellipse of 50-200 m with up to 10% radial jitter.

    The shapes resemble hand-drawn tracks; their splines are smooth simple loops.

    Args:
        draw: Supplied by Hypothesis.
        clockwise: Driving direction; ``None`` draws either.
    """
    count = draw(st.integers(min_value=8, max_value=16))
    width = draw(st.floats(min_value=50, max_value=200))
    height = draw(st.floats(min_value=50, max_value=200))
    jitter = np.array(draw(st.lists(st.floats(-0.1, 0.1), min_size=count, max_size=count)))
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    points = (1 + jitter)[:, None] * np.column_stack(
        [width * np.cos(angles), height * np.sin(angles)]
    )
    if clockwise is None:
        clockwise = draw(st.booleans())
    return points[::-1].copy() if clockwise else points


def track_widths(count: int) -> st.SearchStrategy[FloatArray]:
    """Road widths of 6-20 m, one per control point."""
    return st.lists(st.floats(6.0, 20.0), min_size=count, max_size=count).map(np.array)


@st.composite
def cars_and_actions(draw: st.DrawFn, *, max_steer: float) -> tuple[VehicleState, FloatArray]:
    """1-6 cars anywhere, in any motion up to 120 m/s, with ``[steer, pedal]`` actions.

    Actions reach 1.5, beyond the valid range, so clipping gets exercised too.

    Args:
        draw: Supplied by Hypothesis.
        max_steer: The cars' full steering lock in radians; their wheels are within it.
    """
    count = draw(st.integers(min_value=1, max_value=6))

    def column(low: float, high: float) -> FloatArray:
        return np.array(draw(st.lists(st.floats(low, high), min_size=count, max_size=count)))

    state = VehicleState(
        x=column(-1e3, 1e3),
        y=column(-1e3, 1e3),
        yaw=wrap_angle(column(-np.pi, np.pi)),
        vx=column(0.0, 120.0),
        vy=column(-10.0, 10.0),
        yaw_rate=column(-5.0, 5.0),
        steer=column(-max_steer, max_steer),
    )
    actions = np.column_stack([column(-1.5, 1.5), column(-1.5, 1.5)])
    return state, actions
