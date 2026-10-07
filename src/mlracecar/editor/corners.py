"""Rounding a corner: replacing a sharp point with points along a smooth bend of a chosen radius.

The bend is shaped as roads are: it tightens gradually into a circular arc of the chosen radius
and opens out again (an *easing section* at each end), with the straights on either side
running straight into it. Points are then placed along that shape so that the smooth curve
through them (`mlracecar.core.track.spline`) follows it closely.

Three things make that work, all found by measuring:

- **Easing sections.** The smooth curve can't jump from straight to a circle: its curvature
  changes continuously, so it overshoots, and points placed on a plain circular arc give a
  bend 10-30% tighter than asked. Easing the curvature in and out avoids the jump.
- **Gradually growing gaps.** Where closely spaced points meet a long gap, the curve ripples:
  on a straight, by as much as a bend of 0.4 times the radius. So the new points on each
  straight have gaps that grow by about `GROWTH` from one to the next, from both ends.
- **Points near sharp neighbours too.** A neighbouring corner that's still sharp pulls on the
  curve, by as much as a third of the radius on a gentle bend. Closely spaced points next to
  it keep that pull local. (They also make that corner as sharp as it really is, where a few
  sparse points had let the curve round it off; rounding it too replaces them.)

The straight on each side is rebuilt up to the next point that isn't on it, replacing the
points along it (from rounding other corners, say). So corners can be rounded in any order, and
each still gets its share of the straight.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mlracecar.core.geometry import FloatArray, cross, norm
from mlracecar.core.track.spline import COINCIDENT_GAP
from mlracecar.core.track.validation import DEFAULT_RULES, ValidationRules

EASING_TURN = math.radians(25)
"""How far the road turns in each easing section, at most."""

EASING_SHARE = 1 / 3
"""Share of a corner's turn each easing section may take, so even a slight bend has an arc."""

DOT_TURN = math.radians(15)
"""How far the road turns between neighbouring points of a bend, at most."""

MIN_BEND_GAPS = 6
"""Fewest gaps between the points of a bend, so even a slight bend's shape is traced."""

GROWTH = 3.0
"""How much each gap between points on a straight grows over the one before it."""

MIN_TURN = math.radians(5)
"""A point where the road turns less than this isn't a corner to round. A neighbouring point
that turns at least this much is one, so a bend leaves it half of the straight between them."""

SHARP_TURN = math.radians(30)
"""A neighbouring corner turning at least this much pulls on the curve enough to need closely
spaced points next to it (see `NEAR_CORNER_GAP`). Gentler ones don't: packing points next to
them would only make them sharper."""

NEAR_CORNER_GAP = 1 / 3
"""Next to a sharp neighbouring corner, gaps start at this share of the bend's gap."""

ON_STRAIGHT = 0.002
"""Points within this many metres of a straight count as on it (saved files round to 1 mm)."""

OFF_STRAIGHT = 0.005
"""A bend's first point past the straight is at least this many metres off it, so that later
roundings, walking along the straight, stop where the bend starts. Gentle corners need wider
radii for this: their bends leave the straight very gradually."""

MARGIN = 1.05
"""The smallest radius offered is this much above what the track checks accept, since the
smooth curve comes out a few percent tighter than asked."""

_PATH_STEPS = 2000
"""Steps used to trace the bend's shape, and to space points along a straight."""


class CornerError(ValueError):
    """A corner can't be rounded (to that radius). The message says why, in plain words."""


@dataclass(frozen=True)
class CornerLimits:
    """The radii a corner can be rounded to, in metres."""

    smallest: float
    """The tightest radius the track checks accept for this road width (with a margin)."""
    largest: float
    """The widest radius that fits along the straights on either side."""


@dataclass(frozen=True)
class RoundedCorner:
    """The points of a track with one corner rounded."""

    points: FloatArray
    """All the track's points, shape ``(P, 2)``."""
    widths: FloatArray
    """Road width at each point, shape ``(P,)``."""
    middle: int
    """Index of the bend's middle point, where the corner was."""


def corner_limits(
    points: FloatArray, widths: FloatArray, index: int, rules: ValidationRules = DEFAULT_RULES
) -> CornerLimits:
    """The radii the corner at ``points[index]`` can be rounded to.

    Next to another corner, a bend may use (nearly) half of the straight between them,
    leaving the other half for rounding that corner too. Next to a bend, it may use all of the
    straight up to it, apart from room for a few points between them.

    Raises:
        CornerError: If the point isn't a corner, or no radius fits there.
    """
    corner = _Corner.at(points, index)
    unit = _unit_bend(corner.turn)
    largest = min(
        _largest_for(_Straight.along(points, index, step, corner), unit) for step in (-1, 1)
    )
    width = float(widths[index])
    for_the_road = MARGIN * max(width / 2 + rules.min_inside_radius, rules.min_drivable_radius)
    smallest = float(math.ceil(max(for_the_road, OFF_STRAIGHT / unit.first_offset)))
    if largest < smallest:
        fits = math.floor(largest * 10) / 10  # never shown as enough when it's just short
        raise CornerError(
            f"no radius fits here: the bend would need at least {smallest:.0f} m, but only "
            f"{fits:.1f} m fits along the straights. Move the neighbouring points further "
            "away, or round this corner before the ones next to it."
        )
    return CornerLimits(smallest, largest)


def round_corner(
    points: FloatArray,
    widths: FloatArray,
    index: int,
    radius: float,
    rules: ValidationRules = DEFAULT_RULES,
) -> RoundedCorner:
    """Replace the point at ``index`` with points along a bend of ``radius`` metres.

    The bend touches the straights to the neighbouring corners, and keeps the corner's road
    width. Points along those straights are replaced too, with new ones whose widths follow the
    old ones. Point 0 stays point 0 (it's never replaced), unless it's the corner: then the
    middle of the bend becomes point 0.

    Raises:
        CornerError: If the point isn't a corner, or ``radius`` doesn't fit (the message gives
            the radii that do).
    """
    limits = corner_limits(points, widths, index, rules)
    if not limits.smallest <= radius <= limits.largest:
        raise CornerError(
            f"a {radius:.1f} m radius doesn't fit here: choose between {limits.smallest:.0f} m "
            f"and {limits.largest:.1f} m"
        )
    corner = _Corner.at(points, index)
    unit = _unit_bend(corner.turn)
    before = _Straight.along(points, index, -1, corner)
    after = _Straight.along(points, index, 1, corner)

    # The bend, in the corner's own frame (corner at the origin, arriving along +x, turning
    # left), then turned and mirrored into place.
    local = radius * unit.points - [radius * unit.tangent, 0.0]
    if corner.turn_sign < 0:
        local[:, 1] *= -1
    along = corner.direction_in
    across = np.array([-along[1], along[0]])
    bend = points[index] + local[:, :1] * along + local[:, 1:] * across
    gap = radius * unit.gap
    width = float(widths[index])

    lead_in, lead_in_widths = before.rebuilt(points, widths, bend[0], gap, width)
    lead_out, lead_out_widths = after.rebuilt(points, widths, bend[-1], gap, width)
    new_points = np.vstack([lead_in[::-1], bend, lead_out])
    new_widths = np.concatenate([lead_in_widths[::-1], np.full(len(bend), width), lead_out_widths])

    # Keep everything from the end of the outgoing straight round to the start of the incoming
    # one, then the new points; then turn the loop so point 0 is where it was.
    count = len(points)
    kept = [(after.end + k) % count for k in range((before.end - after.end) % count + 1)]
    result_points = np.vstack([points[kept], new_points])
    result_widths = np.concatenate([widths[kept], new_widths])
    middle = len(kept) + len(lead_in) + len(bend) // 2
    first = middle if index == 0 else kept.index(0)
    return RoundedCorner(
        np.roll(result_points, -first, axis=0),
        np.roll(result_widths, -first),
        (middle - first) % len(result_points),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Corner:
    """The geometry around a point: how the road arrives, leaves, and turns there."""

    direction_in: FloatArray
    """Unit direction from the previous point to this one."""
    direction_out: FloatArray
    """Unit direction from this point to the next one."""
    turn: float
    """How far the road turns here, in radians, from 0 (straight on) towards pi."""
    turn_sign: float
    """+1 turning left, -1 turning right."""

    @classmethod
    def at(cls, points: FloatArray, index: int) -> "_Corner":
        count = len(points)
        if count < 3:
            raise CornerError("a track needs at least 3 points before its corners can be rounded")
        incoming = points[index] - points[(index - 1) % count]
        outgoing = points[(index + 1) % count] - points[index]
        if not min(norm(incoming), norm(outgoing)) >= COINCIDENT_GAP:
            raise CornerError("this point sits on top of its neighbour; move one of them first")
        direction_in, direction_out = incoming / norm(incoming), outgoing / norm(outgoing)
        turn = math.atan2(
            float(cross(direction_in, direction_out)), float(direction_in @ direction_out)
        )
        if abs(turn) < MIN_TURN:
            raise CornerError(
                f"the road barely turns at this point (less than {math.degrees(MIN_TURN):.0f}"
                " degrees): there's no corner to round"
            )
        if abs(turn) > math.pi - MIN_TURN:
            raise CornerError("the road doubles straight back at this point: add a point first")
        return cls(direction_in, direction_out, abs(turn), math.copysign(1.0, turn))


@dataclass(frozen=True)
class _Straight:
    """The straight on one side of a corner: the points along it, up to where it ends."""

    corner: int
    """Index of the corner."""
    on_it: tuple[int, ...]
    """Indices of the points along it, nearest the corner first; the last is where it ends."""
    end: int
    """Index of the point where it ends: a corner, the start of a bend, or point 0."""
    length: float
    """Distance from the corner to the end."""
    leave_half: bool
    """Whether the end is a corner, which needs half the straight to be rounded later."""
    sharp: bool
    """Whether the end is a sharp corner, whose pull needs closely spaced points next to it."""
    far_gap: float
    """Gap between the end and the next point beyond it; infinite if that point is on top of
    the end (a track problem of its own), so the gaps just grow from the bend's side."""

    @classmethod
    def along(cls, points: FloatArray, index: int, step: int, corner: _Corner) -> "_Straight":
        """The straight from ``points[index]`` towards ``index + step`` (``step`` is +-1)."""
        count = len(points)
        origin = points[index]
        direction = corner.direction_out if step > 0 else -corner.direction_in
        on_it = [(index + step) % count]
        reached = float(norm(points[on_it[0]] - origin))
        # Walk on while the next point is further along the same straight. Point 0 (the
        # start/finish line) is never passed, and the walk never comes back round to the corner
        # or onto the other side's straight.
        while on_it[-1] != 0:
            following = (on_it[-1] + step) % count
            if following in (index, (index - step) % count):
                break
            offset = points[following] - origin
            distance = float(offset @ direction)
            if abs(float(cross(direction, offset))) > ON_STRAIGHT or distance <= reached:
                break
            on_it.append(following)
            reached = distance
        end = on_it[-1]
        before_end = points[on_it[-2]] if len(on_it) > 1 else origin
        beyond = points[(end + step) % count]
        turn = _turn(before_end, points[end], beyond)
        far_gap = float(norm(beyond - points[end]))
        return cls(
            index,
            tuple(on_it),
            end,
            reached,
            leave_half=turn >= MIN_TURN,
            sharp=turn >= SHARP_TURN,
            far_gap=far_gap if far_gap >= COINCIDENT_GAP else math.inf,
        )

    def rebuilt(
        self, points: FloatArray, widths: FloatArray, bend_end: FloatArray, gap: float, width: float
    ) -> tuple[FloatArray, FloatArray]:
        """New points (and widths) between the bend's last point on this side, ``bend_end``,
        and the end of the straight, nearest the bend first.

        Widths follow the old points' widths along the straight, starting from the corner's
        ``width`` where the bend ends.
        """
        origin, end = points[self.corner], points[self.end]
        direction = (end - origin) / self.length
        tangent = float((bend_end - origin) @ direction)
        far_gap = min(self.far_gap, NEAR_CORNER_GAP * gap) if self.sharp else self.far_gap
        new = _graded(bend_end, end, near_gap=gap, far_gap=far_gap)
        old = [i for i in self.on_it if float((points[i] - origin) @ direction) > tangent]
        old_distances = [tangent, *(float((points[i] - origin) @ direction) for i in old)]
        old_widths = [width, *(float(widths[i]) for i in old)]
        distances = (new - origin) @ direction
        return new, np.interp(distances, old_distances, old_widths)


def _largest_for(straight: _Straight, unit: "_UnitBend") -> float:
    """The widest radius whose bend fits along one straight.

    Each bend keeps one of its gaps clear of the halfway point next to a corner, and two next
    to another bend (its gap and, about, the other's). Then two neighbouring corners can each
    be rounded as widely as the other, in either order.
    """
    if straight.leave_half:
        return straight.length / 2 / (unit.tangent + unit.gap)
    return straight.length / (unit.tangent + 2 * unit.gap)


def _turn(before: FloatArray, point: FloatArray, after: FloatArray) -> float:
    """How far the road turns at ``point``, in radians (0 if it doubles back on a point)."""
    incoming, outgoing = point - before, after - point
    if not norm(incoming) > 0 or not norm(outgoing) > 0:
        return 0.0
    return abs(math.atan2(float(cross(incoming, outgoing)), float(incoming @ outgoing)))


@dataclass(frozen=True)
class _UnitBend:
    """A bend of radius 1 turning left, from (0, 0) heading along +x."""

    points: FloatArray
    """Its points, evenly spaced along it, shape ``(n, 2)``."""
    tangent: float
    """Distance from the bend's first point to the corner (where the two straights meet)."""
    gap: float
    """Distance between neighbouring points, along the bend."""
    first_offset: float
    """How far the bend's second point is off the incoming straight (the first is on it)."""


@lru_cache(maxsize=64)
def _unit_bend(turn: float) -> _UnitBend:
    """Trace the bend's shape for a radius of 1: curvature eases in, holds, and eases out."""
    easing = min(EASING_TURN, turn * EASING_SHARE)
    easing_length = 2 * easing  # curvature averages half of 1 over an easing section
    length = 2 * easing_length + (turn - 2 * easing)
    s = np.linspace(0.0, length, _PATH_STEPS + 1)
    curvature = np.minimum(
        _smoothstep(s / easing_length), _smoothstep((length - s) / easing_length)
    )
    heading = np.concatenate(([0.0], np.cumsum((curvature[1:] + curvature[:-1]) / 2 * np.diff(s))))
    heading *= turn / heading[-1]  # land exactly on the outgoing direction
    middle = (heading[1:] + heading[:-1]) / 2
    steps = np.column_stack([np.cos(middle), np.sin(middle)]) * np.diff(s)[:, None]
    path = np.vstack([[0.0, 0.0], np.cumsum(steps, axis=0)])

    gaps = max(MIN_BEND_GAPS, math.ceil(turn / DOT_TURN))
    gaps += gaps % 2  # an even number, so a point sits at the middle of the bend
    targets = np.linspace(0.0, length, gaps + 1)
    points = np.column_stack([np.interp(targets, s, path[:, 0]), np.interp(targets, s, path[:, 1])])
    # The outgoing straight leaves the bend's end at angle `turn`; it meets the incoming one
    # (the x axis) at the corner.
    end_x, end_y = path[-1]
    corner_x = end_x - end_y / math.tan(turn)
    return _UnitBend(points, corner_x, length / gaps, float(points[1, 1]))


def _smoothstep(x: FloatArray) -> FloatArray:
    """0 below 0, 1 above 1, and an S-curve between whose slope is 0 at both ends."""
    x = np.clip(x, 0.0, 1.0)
    result: FloatArray = x * x * (3 - 2 * x)
    return result


def _graded(start: FloatArray, end: FloatArray, near_gap: float, far_gap: float) -> FloatArray:
    """Points on the straight from ``start`` towards ``end`` (both excluded), so that each gap
    is at most about `GROWTH` times the one before it, starting from ``near_gap`` next to the
    start and ``far_gap`` next to the end. Shape ``(k, 2)``, in order from the start.

    The wanted gap grows in proportion to the distance from either end, ``gap(x) = near_gap
    + ln(GROWTH) x``: then one gap further along, it's `GROWTH` times bigger. Counting how many
    such gaps fit (``integral dx / gap(x)``), rounding that to a whole number, and spacing the
    points evenly in that count makes them fit the straight exactly.
    """
    distance = float(norm(end - start))  # at least two gaps, by the room each bend leaves
    x = np.linspace(0.0, distance, _PATH_STEPS + 1)
    slope = math.log(GROWTH)
    gap = np.minimum(near_gap + slope * x, far_gap + slope * (distance - x))
    count = np.concatenate(([0.0], np.cumsum(np.diff(x) * (1 / gap[1:] + 1 / gap[:-1]) / 2)))
    gaps = max(1, round(float(count[-1])))
    offsets = np.interp(np.arange(1, gaps) * count[-1] / gaps, count, x)
    direction = (end - start) / distance
    result: FloatArray = start + offsets[:, None] * direction
    return result
