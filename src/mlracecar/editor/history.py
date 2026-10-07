"""Undo and redo for the track editor: the drafts to go back and forward to (ADR-0011).

Drafts are immutable, so undoing an edit means going back to the draft from before it; no edit
needs its own undo code. What counts as one step is up to the caller (the controller): a whole
drag is one step, not one per mouse movement.
"""

from collections import deque
from collections.abc import Hashable
from dataclasses import replace

from mlracecar.editor.draft import TrackDraft

HISTORY_LIMIT = 500
"""How many steps can be undone; older ones are forgotten."""


class History:
    """Earlier drafts to undo back to, and undone ones to redo."""

    def __init__(self, limit: int = HISTORY_LIMIT) -> None:
        if limit < 1:
            raise ValueError(f"the history must hold at least 1 step, got {limit}")
        self._undo: deque[TrackDraft] = deque(maxlen=limit)
        self._redo: list[TrackDraft] = []
        self._merge: Hashable | None = None

    @property
    def can_undo(self) -> bool:
        """Whether there's a step to undo."""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """Whether there's an undone step to redo."""
        return bool(self._redo)

    def record(
        self, before: TrackDraft, after: TrackDraft, *, merge: Hashable | None = None
    ) -> None:
        """Remember that an edit turned ``before`` into ``after``, as one step.

        An edit that changed nothing isn't a step. Edits in a row with the same ``merge`` key
        make one step, so that, say, widening the road at a point notch by notch is undone at
        once. A new step clears the steps that could be redone.
        """
        if before == after:
            return
        if merge is None or merge != self._merge:
            self._undo.append(_kept(before))
        self._merge = merge
        self._redo.clear()

    def undo(self, current: TrackDraft) -> TrackDraft | None:
        """The draft before the last step, or ``None`` if there's nothing to undo.

        ``current`` is the draft being undone; `redo` brings it back.
        """
        if not self._undo:
            return None
        self._redo.append(_kept(current))
        self._merge = None
        return self._undo.pop()

    def redo(self, current: TrackDraft) -> TrackDraft | None:
        """The draft the last undo went back from, or ``None`` if there's nothing to redo."""
        if not self._redo:
            return None
        self._undo.append(_kept(current))
        self._merge = None
        return self._redo.pop()


def _kept(draft: TrackDraft) -> TrackDraft:
    """An equal draft without the cached track and issues, to keep many drafts cheaply.

    A cached track takes about 1.2 MB on a 3.5 km circuit, but only about 13 ms to build again;
    without it, a draft of 100 points is about 2 KB.
    """
    return replace(draft)
