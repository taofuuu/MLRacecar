# ADR-0011: The editor edits immutable drafts

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** tickets #14 (editor), #15 (undo/redo); supersedes the "command pattern" plan in
  [architecture.md §4.12](../architecture.md#412-track-editor)

> **In plain words:** An edit in the track editor never changes the track in place. It makes a
> new copy with the change and leaves the old copy alone. Undo then just means going back to
> the previous copy, which can't go wrong, and every copy can remember its own track checks.

## Context

The architecture planned undo/redo (#15) with the **command pattern**: every edit is an object
with a *do* and an *undo* method. Every kind of edit then needs a hand-written inverse. Get one
subtly wrong (say, deleting a point and then restoring a slightly different width) and the
history quietly corrupts the track.

A track in the editor is small: tens to a few hundred points, each three numbers.

## Options considered

1. **Command objects** with do/undo pairs. The classic approach. Each edit type needs a correct
   inverse, and those inverses need their own tests.
2. **Immutable snapshots.** Every edit returns a new `TrackDraft` and never changes the old one.
   Undo/redo is a list of drafts and a position in it.

## Decision

Use immutable drafts: `TrackDraft` is a frozen dataclass whose edit methods (`append_point`,
`move_point`, `set_width`, ...) return new drafts. Each draft computes its track and its
validation issues once, on first use, and caches them.

## Consequences

- **Positive:** undo/redo (#15) needs no per-edit logic and can't drift from the edits. Edits are
  plain functions, easy to test, including random editing sessions. The view can compare drafts
  to see what changed. Caching per draft means unchanged drafts are never re-validated.
- **Negative / costs:** a copy per edit, which is trivial at this size. Dragging a point creates a
  draft per mouse movement, so the controller must record one history entry per *finished* drag,
  not per movement.
