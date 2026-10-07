"""Tests for mlracecar.editor.history: undo and redo as a list of drafts."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlracecar.editor.draft import TrackDraft
from mlracecar.editor.history import History

SQUARE = TrackDraft(
    points=((100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0), (100.0, -100.0)),
    widths=(12.0,) * 4,
)


def named(number: int) -> TrackDraft:
    """A cheap draft that differs from every other number's."""
    return TrackDraft(name=f"Draft {number}")


def test_a_new_history_has_nothing_to_undo_or_redo() -> None:
    history = History()
    assert not history.can_undo
    assert not history.can_redo
    assert history.undo(named(0)) is None
    assert history.redo(named(0)) is None


def test_undo_goes_back_and_redo_forward() -> None:
    history = History()
    history.record(named(0), named(1))
    assert history.can_undo
    assert history.undo(named(1)) == named(0)
    assert history.can_redo
    assert history.redo(named(0)) == named(1)
    assert not history.can_redo


def test_an_edit_that_changes_nothing_is_not_a_step() -> None:
    history = History()
    history.record(named(0), named(0))
    assert not history.can_undo


def test_a_new_edit_after_undoing_drops_the_undone_steps() -> None:
    history = History()
    history.record(named(0), named(1))
    history.undo(named(1))
    history.record(named(0), named(2))
    assert not history.can_redo
    assert history.undo(named(2)) == named(0)


def test_edits_with_the_same_merge_key_in_a_row_are_one_step() -> None:
    history = History()
    history.record(named(0), named(1), merge="width 3")
    history.record(named(1), named(2), merge="width 3")
    history.record(named(2), named(3), merge="width 5")
    assert history.undo(named(3)) == named(2)
    assert history.undo(named(2)) == named(0)
    assert not history.can_undo


def test_an_undo_ends_a_merge() -> None:
    history = History()
    history.record(named(0), named(1), merge="width 3")
    history.undo(named(1))
    history.redo(named(0))
    history.record(named(1), named(2), merge="width 3")  # a new step, not part of the first
    assert history.undo(named(2)) == named(1)


def test_the_oldest_steps_are_forgotten_beyond_the_limit() -> None:
    history = History(limit=3)
    for number in range(5):
        history.record(named(number), named(number + 1))
    undone = []
    current = named(5)
    while (earlier := history.undo(current)) is not None:
        undone.append(earlier)
        current = earlier
    assert undone == [named(4), named(3), named(2)]


def test_the_limit_must_allow_a_step() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        History(limit=0)


def test_kept_drafts_drop_their_cached_track() -> None:
    # A cached track is about 1.2 MB on a 3.5 km circuit; 500 of them would be too many.
    assert SQUARE.track is not None
    history = History()
    history.record(SQUARE, SQUARE.reverse())
    undone = history.undo(SQUARE.reverse())
    assert undone == SQUARE
    assert "track" not in vars(undone)
    assert undone.track is not None  # built again when needed


# --------------------------------------------------------------------------- #
# A random session, against a timeline of drafts
# --------------------------------------------------------------------------- #

actions = st.lists(st.sampled_from(["edit", "undo", "redo"]), max_size=40)


@given(actions)
def test_any_session_moves_along_a_timeline_of_drafts(session: list[str]) -> None:
    # The model: every draft so far in a list, and which one is current. An edit cuts off
    # whatever could have been redone.
    timeline, position = [named(0)], 0
    history, current = History(), named(0)
    for number, action in enumerate(session, start=1):
        match action:
            case "edit":
                timeline, position = [*timeline[: position + 1], named(number)], position + 1
                history.record(current, named(number))
                current = named(number)
            case "undo":
                earlier = history.undo(current)
                assert (earlier is None) == (position == 0)
                if earlier is not None:
                    position, current = position - 1, earlier
            case "redo":
                later = history.redo(current)
                assert (later is None) == (position == len(timeline) - 1)
                if later is not None:
                    position, current = position + 1, later
        assert current == timeline[position]
        assert history.can_undo == (position > 0)
        assert history.can_redo == (position < len(timeline) - 1)
