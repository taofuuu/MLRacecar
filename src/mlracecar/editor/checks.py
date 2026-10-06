"""Runs the track checks in the background, so the editor never waits for them.

The checks take about 0.1 s on a 3.5 km track: fine after an edit, but too slow for every
frame of a drag. So they run on a worker thread. Drafts are immutable (ADR-0011), which makes
this safe: the worker reads one draft while the user goes on editing and making new ones.

Only the newest draft matters. While a check runs, newer drafts replace each other as "next";
when the check finishes, the next one starts. A drag therefore gets checked a few times a
second, never piling up work. See ADR-0013.
"""

from concurrent.futures import Executor, Future
from typing import NamedTuple

from mlracecar.core.track.validation import ValidationIssue
from mlracecar.editor.draft import TrackDraft


class CheckResult(NamedTuple):
    """The issues found in one draft."""

    draft: TrackDraft
    issues: list[ValidationIssue]


class BackgroundChecks:
    """Checks the newest draft it's given on an executor, one check at a time.

    Call `check` with every draft and `poll` once per frame. Give it a single-thread executor;
    tests can pass one that runs checks immediately.
    """

    def __init__(self, executor: Executor) -> None:
        self._executor = executor
        self._running: tuple[TrackDraft, Future[list[ValidationIssue]]] | None = None
        self._newest: TrackDraft | None = None
        self.result: CheckResult | None = None
        """The most recent finished check, possibly of an older draft than the one on screen."""

    def check(self, draft: TrackDraft) -> None:
        """Ask for ``draft`` to be checked; replaces any draft still waiting its turn."""
        if draft is self._newest:
            return
        self._newest = draft
        if self._running is None:
            self._start(draft)

    def poll(self) -> bool:
        """Collect a finished check, start the next one, and say whether there's a new result."""
        if self._running is None or not self._running[1].done():
            return False
        draft, future = self._running
        self._running = None
        self.result = CheckResult(draft, future.result())
        if self._newest is not None and self._newest is not draft:
            self._start(self._newest)
        return True

    def is_current(self, draft: TrackDraft) -> bool:
        """Whether the latest result is for exactly ``draft``."""
        return self.result is not None and self.result.draft is draft

    def shutdown(self) -> None:
        """Stop the worker, without waiting for a check in progress."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _start(self, draft: TrackDraft) -> None:
        # Reading `issues` caches them on the draft too, so saving it later doesn't re-check.
        self._running = (draft, self._executor.submit(lambda: draft.issues))
