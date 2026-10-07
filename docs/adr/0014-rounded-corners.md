# ADR-0014: How the editor rounds a corner

- **Status:** Accepted
- **Date:** 2026-10-07
- **Related:** ticket #79 (round a corner); ADR-0010 (C2 spline); ADR-0011 (immutable drafts);
  ticket #15 (saved precision)

> **In plain words:** You can sketch a track with one dot per corner, then round each corner to
> a radius you choose. The editor swaps the corner dot for a row of dots along a bend, and the
> dots have to be placed carefully, or the smooth curve drawn through them comes out much
> tighter than you asked. Three tricks, each found by measuring, make the bend come out within
> 6% of the radius: the bend tightens gradually like a real road, the dots on the straights
> are spaced further and further apart, and a few dots sit next to any corner that's still
> sharp.

## Context

The track's centre line is a smooth (C2) curve through the user's dots (ADR-0010). Its
curvature changes continuously and is shaped by every dot, not just the nearest ones. A corner
rounded "by hand", with a few dots on a circular arc, comes out tighter than the arc: that was
already seen in the GP sample (#12), where an 18 m corner came out at 12 m.

The goal (#79): replace a corner dot with dots such that the tightest radius of the actual track
along the bend is within 10% of the radius chosen, for kinks of about 20 degrees up to hairpins of
about 180 degrees, turning either way. A rough sketch rounded at every corner must pass the
track checks.

Measured on a 200 m square and on single corners with 200 m straights (tightest radius on the
bend divided by the radius asked for):

| How the dots are placed | Result |
|---|---|
| On a plain circular arc | 0.23 to 0.77 |
| Arc, plus a dot or two on each straight | 0.86 to 0.88 at best |
| Easing sections, evenly spaced dots | 0.93 to 0.99 with every corner rounded, but down to 0.37 next to a long empty straight |
| Easing sections, gaps growing along the straights | 0.90 to 1.00, but gentle kinks down to 0.3 at their widest radii |
| The same, plus closely spaced dots next to sharp neighbours | **0.94 to 1.00** for every turn and radius offered |

## Options considered

1. **Dots on a circular arc** (a classic fillet). Simple, but the curve can't jump from
   straight to circular: its curvature has to change continuously, so it overshoots. 10-77% too
   tight.
2. **Ease the curvature in and out** (a transition curve, as on real roads and railways), then
   place dots along that shape. The circle in the middle has the radius asked for; the easing
   sections take up to 25 degrees of the turn each, and at most a third of it, so even a slight
   bend has a circular part.
3. **Fix the curve instead**, e.g. a different spline. Out of scope: the curve is shared by the
   simulator, and ADR-0010 chose it for good reasons.

Two further problems showed up with option 2:

- **Ripples where dense dots meet a long gap.** The curve ripples on the straight just before
  the bend, as tight as 0.4 times the radius. Fix: the gaps between new dots on each straight
  grow by about 3 times from one to the next, from both ends.
- **Pull from a neighbouring corner that's still sharp.** It bends a wide, gentle bend by up to
  a third. Fix: next to a sharp neighbour (turning 30 degrees or more), the gaps start at a third
  of the bend's gap, which keeps the pull local.

## Decision

Option 2 with both fixes, in `mlracecar.editor.corners`:

- The bend's dots are evenly spaced along the eased shape, at most 15 degrees of turn apart and at
  least 6 gaps per bend (an even number, so a dot marks the middle of the bend).
- **The straights are rebuilt.** Rounding a corner replaces every dot along each straight, up to
  the next dot that isn't on it (a corner, the start of another bend, or point 0, which is never
  replaced). Widths along the straight follow the old dots' widths.
- **Room:** next to a corner, a bend may use half the straight, less one gap, leaving the other
  half for that corner. Next to a bend, it may use all of the straight but two gaps. Then two
  neighbouring corners get the same room in either order.
- **Smallest radius:** what the track checks accept for the road's width, plus 5%, and wide
  enough that the bend's first dot is at least 5 mm off the straight. Gentle corners leave the
  straight so gradually that their dots would otherwise be mistaken for straight ones.
  Points turning less than 5 degrees aren't corners.
- **Saved files keep millimetres, not centimetres** (changing #15). Centimetre steps, with a
  bend's dots a metre or two apart, shift its tightest radius by up to 9% after reopening (1% for
  millimetres). They would also move the straights' dots up to 7 mm off them, more than the 2 mm
  the rebuilt-straights rule allows.

## Consequences

- **Positive:** bends come out within 6% of the radius asked for (the tests require 10%), from
  10-degree kinks to 165-degree hairpins, turning either way. A rough sketch with every corner
  rounded passes the checks. Rounding all of a sketch's corners takes a few milliseconds.
- **Negative / costs:**
  - **Many dots:** a 90-degree corner adds 11 to 19 dots, most on the straights.
  - **Sharp neighbours show their problems:** a sketch's sparse dots let the curve round off its
    corners generously. Once a neighbour's rounding packs dots next to a corner, that corner's
    real sharpness shows, often as a red problem marker, until it's rounded too.
  - **Order can matter where space is tight:** where three sharp corners sit close together,
    rounding the outer two first can leave the middle one too little room. The message says so,
    and undo lets the user round them in another order.
  - **Rules of thumb in the code:** the constants (25 and 15 degrees, growth of 3, a third, 5 mm,
    2 mm) were tuned by the measurements above, not derived.
- **Follow-ups:** #16 (user guide) explains sketching one dot per corner and rounding with hold C
  and the wheel.
