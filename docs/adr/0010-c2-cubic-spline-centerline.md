# ADR-0010: Smooth-bend (C2) cubic spline for the track centerline

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** supersedes the curve type in [ADR-0004](0004-track-representation.md); ticket #10

> **In plain words:** We switched the curve that turns your clicked dots into a track. The old
> choice kept the road's *direction* smooth but let its *bend* jump at every dot, so a corner
> drawn as one steady arc would tighten and loosen as you drove through it. The new curve keeps
> the bend smooth too. The cost is that unevenly spaced dots can make it bulge into a small
> loop more often; the track checks (#12) catch that, and the editor will show it live.

## Context

ADR-0004 chose a closed **centripetal Catmull-Rom** spline: it passes through every control
point, cannot form cusps or loops within a piece, and changing a point only affects its
neighbouring pieces. Its direction is continuous (C1), but its curvature is not: it jumps at
every control point.

Building #10 showed how much that matters. Curvature feeds the agent's "road ahead"
observations (M3-2), and drives how a corner feels. Measured on a 50 m circle:

| Control points per full circle | Catmull-Rom (C1) curvature error | C2 cubic curvature error |
|--------------------------------|----------------------------------|--------------------------|
| 8 (a dot every ~39 m)          | 51.5%                            | 5.7%                     |
| 12                             | 21.5%                            | 2.4%                     |
| 16                             | 11.9%                            | 1.3%                     |
| 20                             | n/a                              | 0.8%                     |

Averaging Catmull-Rom's bend over 10 m of road still leaves 30% error with 8 points. Hand-drawn
tracks typically have a point every 20–40 m, so corners would bend unevenly by tens of percent.

## Options considered

1. **Keep centripetal Catmull-Rom (C1).** Most robust and most local; curvature wobbles at every
   control point.
2. **Closed C2 cubic interpolating spline with centripetal knot spacing.** Passes through every
   point; position, direction, and curvature are all continuous; about 10× more accurate
   curvature. It's global (moving one point shifts the others slightly, an effect that fades
   about 4× per point away), and overshoots more on uneven input: on 300 random, deliberately
   uneven star-shaped point sets, the centerline crossed itself 90 times versus 29 for
   Catmull-Rom.
3. **Closed cubic B-spline (approximating).** C2, local, and never overshoots its control
   polygon, but the curve no longer passes through the clicked points, which makes editing
   feel indirect.

## Decision

Use option 2: a **closed C2 cubic interpolating spline** with centripetal (square-root of
chord length) knot spacing. The coefficients come from one small cyclic linear system per
track, solved with NumPy (the system is strictly diagonally dominant, so it always has a
unique solution).

## Consequences

- **Positive:** corners drawn as steady arcs bend steadily; curvature is continuous everywhere,
  so observations and validation see the track's real shape. As a bonus, the shape itself fits
  intended arcs more closely (radius error 0.12% vs 0.85% at 8 points per circle).
- **Negative / costs:** loops on uneven input are more likely, so the validation rules (#12) must
  check the centerline and boundaries for self-intersection (already planned), and the editor
  must show problems live (#14). Moving a point nudges its neighbours slightly.
- **Unchanged from ADR-0004:** tracks are still stored as control points with widths in
  versioned JSON, and everything else is derived.
