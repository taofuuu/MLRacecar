# ADR-0004: Tracks as closed splines with a width profile, stored as versioned JSON

- **Status:** Accepted; curve type superseded by [ADR-0010](0010-c2-cubic-spline-centerline.md)
- **Date:** 2026-10-06
- **Related:** tickets M1-2 to M1-6

> **Superseded in part:** the curve type below (centripetal Catmull-Rom) was replaced by a C2
> cubic spline in [ADR-0010](0010-c2-cubic-spline-centerline.md). Storing tracks as control
> points with widths in versioned JSON, with everything else derived, still stands.

> **In plain words:** A track is saved as the list of dots you click, each with a road width. The program draws a smooth curve through the dots and calculates everything else (edges, checkpoints, starting positions) from them. Files stay small and readable, and each one records its format version so old tracks keep working.

## Context

Users design tracks visually. The simulation needs precise geometry for progress tracking
(arc length along the track), off-track detection, raycasts, and checkpoints. Track files
should be small, diffable in git, and stay loadable as the format evolves.

## Options considered

1. **Closed spline through control points, with per-point width.** Compact, precise, and
   intuitive to edit. Branching layouts (pit lanes) aren't possible without a format change.
2. **Bitmap mask (paint the track).** Easy to sketch, but imprecise, with no natural notion
   of "progress", and expensive raycasts.
3. **Explicit left/right boundary polylines.** Precise, but tedious to edit and easy to make
   inconsistent.
4. **Tile-based pieces (like a toy race set).** Easy to validate, but very limiting creatively.

## Decision

A track is an ordered list of control points `(x, y, width)` defining a **closed centripetal
Catmull-Rom spline**:

- It passes through every control point, so editing is intuitive.
- The centripetal parameterization avoids cusps and self-intersections within a segment.
- It is C1-continuous, so there are no kinks in the tangent and the normals are stable.

All other geometry (resampled centerline, boundaries, checkpoints, start grid) is **derived,
never stored**. Files are JSON with a `schema_version` and a migration registry.

## Consequences

- **Positive:** a single source of truth; small, diffable files; exact geometry for race
  rules and sensors.
- **Negative / costs:** no branching layouts or surface types in v1, which would need
  schema v2. Validation must catch boundary folding on tight corners
  (corner radius < half the track width).
