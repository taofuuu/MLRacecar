"""Realistic test circuits, laid out the way real circuits are designed.

A layout is a sequence of straights and constant-radius corners with real-world sizes (slow
hairpins ~12 m radius, chicanes ~20 m, medium corners 35-60 m, fast sweepers 100-220 m).
Two straights in different directions get their lengths solved so the lap closes exactly.
Control points are then placed the way a person would in the editor: sparse on straights,
denser in corners, one where every element starts.
"""

from typing import NamedTuple

import numpy as np

from mlracecar.core.geometry import FloatArray

GP_ROAD_WIDTH = 14.0  # typical F1 track width, metres


class Straight(NamedTuple):
    length: float  # metres; ignored for the two straights solved to close the lap
    name: str = ""
    solved: bool = False


class Corner(NamedTuple):
    name: str
    angle: float  # degrees, + left / - right
    radius: float  # metres, at the centerline


type Element = Straight | Corner


class Circuit(NamedTuple):
    points: FloatArray
    """Control points, shape ``(P, 2)``."""
    widths: FloatArray
    """Road width at each control point, shape ``(P,)``."""
    corners: dict[str, list[int]]
    """Control-point indices placed in each named corner."""


def gp_layout(chicane_radius: float = 20.0) -> list[Element]:
    """A 3.5 km clockwise GP-style circuit: 13 corners, from a 12 m hairpin to a 220 m sweeper."""
    return [
        Straight(0, "main straight", solved=True),  # heading east
        Corner("T1", -90, 18),  # heavy braking at the end of the main straight
        Straight(100),
        Corner("T2", 50, 45),
        Straight(120),
        Corner("T3", -50, 110),  # fast esses
        Corner("T4", -40, 100),
        Corner("T5", 40, 100),
        Straight(180),
        Corner("T6", -90, 38),  # onto the back straight
        Straight(600, "back straight"),
        Corner("T7", 40, chicane_radius),  # bus-stop chicane
        Straight(20),
        Corner("T8", -40, chicane_radius),
        Straight(200),
        Corner("T9", -45, 55),  # double-apex right
        Straight(40),
        Corner("T9b", -45, 55),
        Straight(0, "north straight", solved=True),
        Corner("T10", 35, 220),  # flat-out sweeper
        Straight(120),
        Corner("T11", -165, 12),  # slowest hairpin
        Straight(220),
        Corner("T12", 85, 45),
        Straight(90),
        Corner("T13", -45, 70),  # final corner
    ]


def gp_circuit(variant: str = "clean") -> Circuit:
    """The GP circuit, 14 m wide.

    ``"clean"``: realistic sizes throughout.
    ``"limit"``: pushed past what a road can do: a 7.5 m chicane, the T11 hairpin widened to
    24 m, and T1 narrowed to 5 m.
    """
    limit = variant == "limit"
    circuit = build_circuit(gp_layout(chicane_radius=7.5 if limit else 20.0), GP_ROAD_WIDTH)
    if limit:
        circuit.widths[circuit.corners["T11"]] = 24.0
        circuit.widths[circuit.corners["T1"]] = 5.0
    return circuit


def build_circuit(layout: list[Element], width: float) -> Circuit:
    """Trace a layout, close the lap, and place control points along it."""
    lengths = _closing_lengths(layout)
    position, heading = np.zeros(2), 0.0
    points: list[FloatArray] = []
    corners: dict[str, list[int]] = {}
    for element in layout:
        if isinstance(element, Straight):
            length = lengths.get(id(element), element.length)
            count = max(1, int(np.ceil(length / 60)))  # a dot every 60 m or so
            direction = np.array([np.cos(heading), np.sin(heading)])
            points += [position + k * length / count * direction for k in range(count)]
            position = position + length * direction
        else:
            angle = np.radians(element.angle)
            # a dot every 25 m, and at least every 0.4 rad (23 degrees) of turning
            count = max(2, int(np.ceil(max(abs(angle) * element.radius / 25, abs(angle) / 0.4))))
            corners[element.name] = list(range(len(points), len(points) + count))
            points += [_arc(position, heading, element, k / count) for k in range(count)]
            position, heading = _arc(position, heading, element, 1.0), heading + angle
    array = np.round(np.array(points), 1)
    return Circuit(array, np.full(len(array), width), corners)


def _arc(position: FloatArray, heading: float, corner: Corner, fraction: float) -> FloatArray:
    """The point ``fraction`` of the way around a corner that starts at ``position``."""
    side = np.sign(corner.angle)
    center = position + corner.radius * side * np.array([-np.sin(heading), np.cos(heading)])
    turned = heading + np.radians(corner.angle) * fraction
    result: FloatArray = center - corner.radius * side * np.array([-np.sin(turned), np.cos(turned)])
    return result


def _end(layout: list[Element], lengths: dict[int, float]) -> FloatArray:
    """Where the centerline ends, given lengths for the solved straights."""
    position, heading = np.zeros(2), 0.0
    for element in layout:
        if isinstance(element, Straight):
            length = lengths.get(id(element), element.length)
            position = position + length * np.array([np.cos(heading), np.sin(heading)])
        else:
            position, heading = (
                _arc(position, heading, element, 1.0),
                heading + np.radians(element.angle),
            )
    return position


def _closing_lengths(layout: list[Element]) -> dict[int, float]:
    """Lengths of the two solved straights that bring the lap back to its start."""
    first, second = (id(e) for e in layout if isinstance(e, Straight) and e.solved)
    base = _end(layout, {first: 0.0, second: 0.0})
    along_first = _end(layout, {first: 1.0, second: 0.0}) - base
    along_second = _end(layout, {first: 0.0, second: 1.0}) - base
    lengths = np.linalg.solve(np.column_stack([along_first, along_second]), -base)
    return {first: float(lengths[0]), second: float(lengths[1])}
