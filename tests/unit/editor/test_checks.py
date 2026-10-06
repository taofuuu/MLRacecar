"""Tests for mlracecar.editor.checks: running the track checks in the background."""

import threading
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any

from executors import InlineExecutor
from mlracecar.core.track.validation import IssueCode
from mlracecar.editor.checks import BackgroundChecks
from mlracecar.editor.draft import TrackDraft

SQUARE = TrackDraft(
    points=((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0)),
    widths=(12.0,) * 4,
)


class ManualExecutor(Executor):
    """Holds submitted work until the test runs it, to control when checks finish."""

    def __init__(self) -> None:
        self.waiting: list[tuple[Future[Any], Any]] = []

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        self.waiting.append((future, lambda: fn(*args, **kwargs)))
        return future

    def finish_next(self) -> None:
        future, work = self.waiting.pop(0)
        future.set_result(work())


def test_a_check_reports_the_issues_of_its_draft() -> None:
    checks = BackgroundChecks(InlineExecutor())
    narrow = SQUARE.set_width(2, 3.0)
    checks.check(narrow)
    assert checks.poll()
    assert checks.result is not None
    assert checks.result.draft is narrow
    assert [issue.code for issue in checks.result.issues] == [IssueCode.TOO_NARROW]
    assert checks.is_current(narrow)
    assert not checks.is_current(SQUARE)


def test_nothing_new_until_a_check_finishes() -> None:
    executor = ManualExecutor()
    checks = BackgroundChecks(executor)
    assert not checks.poll()  # nothing asked for yet
    checks.check(SQUARE)
    assert not checks.poll()  # still running
    executor.finish_next()
    assert checks.poll()
    assert not checks.poll()  # already collected


def test_asking_again_for_the_same_draft_does_nothing() -> None:
    executor = InlineExecutor()
    checks = BackgroundChecks(executor)
    checks.check(SQUARE)
    checks.check(SQUARE)
    assert executor.submitted == 1


def test_while_a_check_runs_only_the_newest_draft_waits_its_turn() -> None:
    executor = ManualExecutor()
    checks = BackgroundChecks(executor)
    drafts = [SQUARE.set_width(0, width) for width in (10.0, 11.0, 12.0, 13.0)]
    for draft in drafts:  # as during a drag: a new draft every frame
        checks.check(draft)
    assert len(executor.waiting) == 1  # the first one is running; the rest just queue up
    executor.finish_next()
    assert checks.poll()
    assert checks.result is not None
    assert checks.result.draft is drafts[0]
    assert len(executor.waiting) == 1  # the newest draft started; the two in between never ran
    executor.finish_next()
    assert checks.poll()
    assert checks.is_current(drafts[-1])
    assert executor.waiting == []


def test_checks_run_on_another_thread_and_do_not_block() -> None:
    started, release = threading.Event(), threading.Event()

    class SlowDraft(TrackDraft):
        @property
        def issues(self) -> list[Any]:
            started.set()
            release.wait(timeout=5)
            return []

    checks = BackgroundChecks(ThreadPoolExecutor(max_workers=1))
    slow = SlowDraft(points=SQUARE.points, widths=SQUARE.widths)
    begin = time.perf_counter()
    checks.check(slow)
    assert started.wait(timeout=5)
    assert time.perf_counter() - begin < 1  # `check` returned while the check was running
    assert not checks.poll()
    release.set()
    deadline = time.perf_counter() + 5
    while not checks.poll():
        assert time.perf_counter() < deadline
        time.sleep(0.01)
    assert checks.is_current(slow)
    checks.shutdown()
