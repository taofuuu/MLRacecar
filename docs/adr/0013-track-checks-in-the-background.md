# ADR-0013: The editor runs the track checks in the background

- **Status:** Accepted
- **Date:** 2026-10-07
- **Related:** ticket #14 (track editor); ADR-0011 (immutable drafts); ticket #12 (validation)

> **In plain words:** Checking a big track for problems takes about a tenth of a second. That's
> fine after one click, but far too slow to do for every frame while you drag a point: the
> editor would stutter. So the checks run on a second "worker" while the window keeps drawing.
> The problem markers catch up a few times a second during a drag and are exact once you let go.

## Context

The editor highlights track problems as you edit (#14). The checks (`validate`, #12) take
**about 110 ms on the 3.5 km GP sample track**; about 90% of that is the search for places where
the road's edges cross. Dragging a point makes a new draft every frame. Checking each one
before drawing it would make frames about 130 ms long, about 8 frames per second.

## Options considered

1. **Check every draft before drawing it.** Simple, but the editor stutters on big tracks.
2. **Check only when the user pauses** (say 0.2 s after the last change). Smooth, but there's
   no feedback during a drag, which is when you'd want to watch a bend fold or unfold.
3. **Check on a worker thread, newest draft first.** Smooth, *and* the markers update during
   a drag. Needs care: two threads touching shared data.
4. **Make the checks faster.** Worth doing anyway, but even a fast check takes time that grows
   with the track's length, and the editor shouldn't depend on it.

## Decision

Option 3. `BackgroundChecks` (`mlracecar.editor.checks`) runs one check at a time on a
single-thread executor. While a check runs, newer drafts replace each other as the next in
line, so a drag never piles up work: when a check finishes, only the newest draft is checked
next.

It's safe because drafts are immutable (ADR-0011): the worker reads a draft that nothing can
change, while the window carries on with new drafts. No locks are needed.

The window shows the latest results even if they're for a slightly older draft, as long as it
has the same number of points. After a point is added or deleted, point numbers in old results
would be wrong, so the window shows "checking..." until the new results arrive.

## Consequences

- **Positive:** dragging on the GP track takes about **17 ms per frame** (60 fps), with about
  3 check results a second during the drag. An idle editor still uses no CPU. A draft's
  results stay cached on it, so saving a draft that has already been checked costs nothing.
- **Negative / costs:** during a drag, the markers lag the road by up to about 0.3 s. Python
  runs only one thread at a time (the GIL), so a check slows drawing slightly while it runs.
  Tests swap the thread for an executor that runs checks immediately, which keeps them
  deterministic; one test uses a real thread.
- **Follow-ups:** speed up the edge-crossing search (#80).
