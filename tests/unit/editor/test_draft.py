"""Tests for mlracecar.editor.draft: the editor's headless model."""

import contextlib
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from circuits import gp_circuit
from mlracecar.core.track.validation import has_errors
from mlracecar.editor.corners import CornerError
from mlracecar.editor.draft import DEFAULT_WIDTH, MAX_WIDTH, MIN_WIDTH, Point, TrackDraft
from mlracecar.io.track_file import (
    TrackFileError,
    format_track_file,
    parse_track_file,
    read_track_file,
)

SAMPLES = Path(__file__).parents[3] / "tracks"


def oval_clicks(count: int = 12) -> list[Point]:
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return [(float(120 * np.cos(a)), float(60 * np.sin(a))) for a in angles]


def clicked(points: list[Point]) -> TrackDraft:
    draft = TrackDraft()
    for point in points:
        draft = draft.append_point(point)
    return draft


def signed_area(points: tuple[Point, ...]) -> float:
    xy = np.array(points)
    return float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]) / 2)


# --------------------------------------------------------------------------- #
# Building a track from scratch
# --------------------------------------------------------------------------- #


def test_clicking_around_an_oval_builds_a_raceable_track() -> None:
    draft = clicked(oval_clicks())
    assert draft.points == tuple(oval_clicks())  # clicked in order, so they stay in order
    assert draft.widths == (DEFAULT_WIDTH,) * 12
    assert draft.issues == []
    assert draft.is_raceable
    saved = parse_track_file(format_track_file(draft.to_track_file()))
    assert TrackDraft.from_track_file(saved) == draft


def test_clicking_a_whole_gp_circuit_in_order_keeps_every_point_in_place() -> None:
    circuit = gp_circuit("clean")
    draft = clicked([(float(x), float(y)) for x, y in circuit.points])
    np.testing.assert_array_equal(draft.points, circuit.points)
    assert draft.is_raceable


def test_a_new_draft_explains_what_is_missing() -> None:
    draft = TrackDraft().append_point((0.0, 0.0)).append_point((50.0, 0.0))
    assert draft.track is None
    assert not draft.is_raceable
    assert "at least 4 points" in draft.issues[0].message
    with pytest.raises(TrackFileError, match="at least 3 control points"):
        draft.to_track_file()


def test_first_point_gets_the_default_width_and_later_ones_inherit() -> None:
    draft = TrackDraft().append_point((0.0, 0.0)).change_width(0, 3.0).append_point((50.0, 0.0))
    assert draft.widths == (DEFAULT_WIDTH + 3.0, DEFAULT_WIDTH + 3.0)


# --------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------- #


def test_inserting_with_too_few_points_appends() -> None:
    draft = TrackDraft().insert_point((0.0, 0.0)).insert_point((10.0, 0.0))
    assert draft.points == ((0.0, 0.0), (10.0, 0.0))


def test_a_click_beside_the_road_is_inserted_into_that_stretch() -> None:
    draft = clicked(oval_clicks())
    inserted = draft.insert_point((85.0, 45.0))  # by the road from point 1 (104, 30) to 2 (60, 52)
    assert len(inserted.points) == 13
    assert inserted.points[2] == (85.0, 45.0)
    assert inserted.points[1] == draft.points[1]
    assert inserted.points[3] == draft.points[2]


def test_an_inserted_point_takes_the_road_width_at_that_spot() -> None:
    draft = clicked(oval_clicks()).set_width(1, 20.0).set_width(2, 20.0)
    inserted = draft.insert_point((85.0, 45.0))  # between the two 20 m points
    assert inserted.widths[2] == pytest.approx(20.0)


def test_editing_never_changes_the_original() -> None:
    draft = clicked(oval_clicks())
    before = (draft.points, draft.widths)
    draft.move_point(3, (0.0, 0.0))
    draft.delete_point(5)
    draft.set_width(2, 30.0)
    draft.reverse()
    draft.set_start(4)
    draft.append_point((0.0, 70.0))
    draft.insert_point((0.0, 62.0))
    assert (draft.points, draft.widths) == before


def test_move_and_delete() -> None:
    draft = clicked(oval_clicks())
    moved = draft.move_point(3, (1.0, 2.0))
    assert moved.points[3] == (1.0, 2.0)
    deleted = draft.delete_point(3)
    assert len(deleted.points) == 11
    assert deleted.points[3] == draft.points[4]


@pytest.mark.parametrize(
    ("width", "expected"), [(25.0, 25.0), (0.2, MIN_WIDTH), (500.0, MAX_WIDTH)]
)
def test_widths_stay_within_limits(width: float, expected: float) -> None:
    assert clicked(oval_clicks()).set_width(0, width).widths[0] == expected


def test_change_width_adds_to_the_current_width() -> None:
    assert clicked(oval_clicks()).change_width(4, -2.5).widths[4] == DEFAULT_WIDTH - 2.5


def test_reverse_drives_the_other_way_from_the_same_start() -> None:
    draft = clicked(oval_clicks())
    reversed_draft = draft.reverse()
    assert reversed_draft.points[0] == draft.points[0]
    assert reversed_draft.points[1] == draft.points[-1]
    assert math.copysign(1, signed_area(reversed_draft.points)) == -math.copysign(
        1, signed_area(draft.points)
    )
    assert reversed_draft.reverse() == draft


def test_set_start_moves_the_start_line_and_keeps_the_direction() -> None:
    draft = clicked(oval_clicks())
    moved = draft.set_start(4)
    assert moved.points[0] == draft.points[4]
    assert moved.points[1] == draft.points[5]
    assert moved.set_start(len(draft.points) - 4) == draft


def test_rename() -> None:
    draft = TrackDraft()
    assert draft.rename("  Monza-ish ").name == "Monza-ish"
    assert draft.rename("   ").name == draft.name  # an empty name keeps the old one


def test_round_corner_gives_the_new_draft_and_the_middle_of_the_bend() -> None:
    square = TrackDraft(
        points=((100.0, -100.0), (100.0, 100.0), (-100.0, 100.0), (-100.0, -100.0)),
        widths=(12.0,) * 4,
        name="Square",
    )
    limits = square.corner_limits(1)
    rounded, middle = square.round_corner(1, 30.0)
    assert limits.smallest <= 30.0 <= limits.largest
    assert rounded.name == "Square"
    assert len(rounded.points) == len(rounded.widths) > 4
    assert all(type(x) is float and type(y) is float for x, y in rounded.points)  # not numpy
    distances = [math.dist(point, (100.0, 100.0)) for point in rounded.points]
    assert middle == distances.index(min(distances))
    assert rounded.track is not None


def test_round_corner_checks_the_point_exists() -> None:
    with pytest.raises(IndexError):
        TrackDraft().corner_limits(0)
    with pytest.raises(IndexError):
        TrackDraft().round_corner(0, 10.0)


def test_rounding_tidies_positions_and_widths() -> None:
    draft = TrackDraft(points=((1.23456, -0.004), (-314.0837535325377, 7.0)), widths=(12.378, 9.0))
    rounded = draft.rounded(2)
    assert rounded == TrackDraft(points=((1.23, 0.0), (-314.08, 7.0)), widths=(12.38, 9.0))
    assert math.copysign(1.0, rounded.points[0][1]) == 1.0  # 0.0, not -0.0


MISSING_POINT_EDITS = {
    "move": lambda draft: draft.move_point(4, (0.0, 0.0)),
    "delete": lambda draft: draft.delete_point(4),
    "width": lambda draft: draft.set_width(4, 10.0),
    "start": lambda draft: draft.set_start(4),
}


@pytest.mark.parametrize("edit", MISSING_POINT_EDITS)
def test_editing_a_missing_point_is_an_error(edit: str) -> None:
    draft = clicked(oval_clicks(4))
    with pytest.raises(IndexError, match="no point 4; the draft has 4 points"):
        MISSING_POINT_EDITS[edit](draft)


def test_points_and_widths_must_match() -> None:
    with pytest.raises(ValueError, match="2 points but 1 widths"):
        TrackDraft(points=((0.0, 0.0), (1.0, 0.0)), widths=(5.0,))


# --------------------------------------------------------------------------- #
# Finding points and stretches
# --------------------------------------------------------------------------- #


def test_point_near_finds_the_closest_point_within_the_radius() -> None:
    draft = clicked(oval_clicks())
    assert draft.point_near((119.0, 1.0), radius=2.0) == 0
    assert draft.point_near((115.0, 0.0), radius=2.0) is None
    assert TrackDraft().point_near((0.0, 0.0), radius=100.0) is None


@pytest.mark.parametrize(
    ("position", "on_road"),
    [((120.0, 0.0), True), ((125.5, 0.0), True), ((126.5, 0.0), False), ((0.0, 0.0), False)],
)
def test_is_on_road(position: Point, on_road: bool) -> None:
    draft = clicked(oval_clicks())  # 12 m wide: 6 m either side of the middle at (120, 0)
    assert draft.is_on_road(position) is on_road


def test_nothing_is_on_the_road_until_there_is_a_track() -> None:
    assert not TrackDraft().append_point((0.0, 0.0)).is_on_road((0.0, 0.0))


def test_adding_to_a_broken_draft_uses_the_straight_lines_between_points() -> None:
    # Points 1 and 2 coincide, so there's no curve yet; the click still finds its stretch.
    draft = TrackDraft(
        points=((0.0, 0.0), (100.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
        widths=(10.0, 10.0, 14.0, 10.0, 10.0),
    )
    assert draft.track is None
    inserted = draft.insert_point((101.0, 50.0))  # beside the line from point 2 to point 3
    assert inserted.points[3] == (101.0, 50.0)
    assert inserted.widths[3] == 12.0  # halfway between the two widths around it


# --------------------------------------------------------------------------- #
# Derived results and files
# --------------------------------------------------------------------------- #


def test_derived_results_are_computed_once_per_draft() -> None:
    draft = clicked(oval_clicks())
    assert draft.issues is draft.issues
    assert draft.track is draft.track


def test_a_draft_with_errors_can_still_be_saved() -> None:
    draft = clicked(oval_clicks()).set_width(2, 3.0)  # too narrow: an error, not a crash
    assert not draft.is_raceable
    assert draft.to_track_file().control_points[2].width == 3.0


@pytest.mark.parametrize("name", ["oval", "gp-circuit"])
def test_sample_tracks_open_and_save_unchanged(name: str) -> None:
    track_file = read_track_file(SAMPLES / f"{name}.json")
    draft = TrackDraft.from_track_file(track_file)
    assert draft.is_raceable
    assert draft.to_track_file() == track_file


def test_metadata_is_kept() -> None:
    track_file = (
        clicked(oval_clicks())
        .to_track_file()
        .model_copy(update={"metadata": {"editor": {"zoom": 2.0}}})
    )
    assert TrackDraft.from_track_file(track_file).to_track_file().metadata == {
        "editor": {"zoom": 2.0}
    }


# --------------------------------------------------------------------------- #
# A random editing session
# --------------------------------------------------------------------------- #

coordinate = st.floats(min_value=-500, max_value=500)
edits = st.lists(
    st.one_of(
        st.tuples(st.just("append"), coordinate, coordinate),
        st.tuples(st.just("insert"), coordinate, coordinate),
        st.tuples(st.just("move"), st.integers(0, 50), coordinate, coordinate),
        st.tuples(st.just("delete"), st.integers(0, 50)),
        st.tuples(st.just("width"), st.integers(0, 50), st.floats(-30, 30)),
        st.tuples(st.just("reverse")),
        st.tuples(st.just("start"), st.integers(0, 50)),
        st.tuples(st.just("round"), st.integers(0, 50), st.floats(5, 100)),
    ),
    max_size=25,
)


@given(edits)
def test_any_editing_session_keeps_the_draft_consistent(session: list[tuple[object, ...]]) -> None:
    draft = clicked(oval_clicks(6))
    for edit in session:
        count = len(draft.points)
        match edit:
            case ("append", float(x), float(y)):
                draft = draft.append_point((x, y))
            case ("insert", float(x), float(y)):
                draft = draft.insert_point((x, y))
            case ("move", int(i), float(x), float(y)) if count:
                draft = draft.move_point(i % count, (x, y))
            case ("delete", int(i)) if count:
                draft = draft.delete_point(i % count)
            case ("width", int(i), float(by)) if count:
                draft = draft.change_width(i % count, by)
            case ("reverse",):
                draft = draft.reverse()
            case ("start", int(i)) if count:
                draft = draft.set_start(i % count)
            case ("round", int(i), float(radius)) if count:
                # Not a corner, or the radius doesn't fit: then the draft stays as it was.
                with contextlib.suppress(CornerError):
                    draft, _ = draft.round_corner(i % count, radius)
    assert len(draft.points) == len(draft.widths)
    assert all(MIN_WIDTH <= width <= MAX_WIDTH for width in draft.widths)
    assert isinstance(draft.issues, list)  # validation copes with whatever the session made
    assert draft.is_raceable == (draft.track is not None and not has_errors(draft.issues))
