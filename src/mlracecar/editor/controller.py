"""Turns mouse and keyboard input into draft edits and camera moves.

The controller takes positions in window pixels and converts them with the camera, so every
interaction can be tested without a window: the pygame window (`mlracecar.editor.app`) only
translates its events into these calls.

Mouse:

- **Click on the road** inserts a point into that stretch; hold and drag to place it.
- **Click anywhere else** adds a point after the last one, so clicking around in order draws a
  track. **Shift+click** always adds after the last one, even on the road.
- **Drag** a point to move it. A click on a point without dragging selects it.
- **Right-click** a point to delete it.
- **Right-drag** or **middle-drag** pans; the **wheel** zooms around the cursor.
- **Shift+wheel** widens or narrows the road at the point under the cursor (or the selected one).
- **Hold C and turn the wheel** to round the corner under the cursor (or the selected one): the
  wheel changes the radius, the rounded corner shows as you go, and letting go of C keeps it.

Every edit can be undone (`EditorController.undo`). One undo step is one whole action: a click,
or a drag with the point the click may have added. Widening or narrowing the road at a point
notch by notch, with the wheel or the keys, is one step too, and so is rounding a corner,
however much the wheel turned.
"""

from collections.abc import Hashable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mlracecar.editor.corners import CornerError, CornerLimits
from mlracecar.editor.draft import Point, TrackDraft
from mlracecar.editor.history import History
from mlracecar.render.camera import Camera

type Pixel = tuple[float, float]

GRAB_RADIUS = 10.0
"""How close to a point, in pixels, the cursor must be to grab it."""

DRAG_THRESHOLD = 3.0
"""Pixels the mouse must move while held before a click becomes a drag."""

WIDTH_STEP = 1.0
"""Metres of road width per wheel notch or key press."""

ZOOM_STEP = 1.2
"""Zoom factor per wheel notch or key press."""

PAN_STEP = 60.0
"""Pixels per arrow key press."""

DEFAULT_RADIUS = 25.0
"""Radius, in metres, the first corner is rounded to; after that, the one used last."""

RADIUS_STEP = 1.1
"""Factor the corner radius changes by per wheel notch."""


class Button(IntEnum):
    """Mouse buttons, numbered as pygame and SDL number them."""

    LEFT = 1
    MIDDLE = 2
    RIGHT = 3


@dataclass
class _PointDrag:
    before: TrackDraft
    """The draft before the press, so the whole gesture is one undo step."""
    index: int
    offset: Point
    """From the cursor to the point at the press, in metres, so the point doesn't jump."""
    pressed_at: Pixel
    moving: bool = False


@dataclass
class _Pan:
    button: Button
    last: Pixel


@dataclass
class _Rounding:
    before: TrackDraft
    """The draft before rounding started, so the whole rounding is one undo step."""
    index: int
    """The corner being rounded, in ``before``."""
    limits: CornerLimits
    wanted: float
    """The radius the wheel has asked for, before rounding it to whole metres."""

    @property
    def radius(self) -> float:
        """The radius used: ``wanted`` in whole metres, within the limits."""
        return float(np.clip(round(self.wanted), self.limits.smallest, self.limits.largest))


@dataclass(frozen=True)
class Rounding:
    """The corner being rounded, for the status bar."""

    point: int
    """Index of the corner in the track before rounding."""
    radius: float
    """Metres."""
    limits: CornerLimits


class EditorController:
    """The editor's state: the draft, the camera, and what the user is pointing at or holding.

    The draft itself is immutable (ADR-0011); the controller swaps in a new one on every edit,
    and keeps the earlier ones in `history` for undo.
    """

    def __init__(self, draft: TrackDraft, camera: Camera, *, snap: bool = False) -> None:
        self.draft = draft
        self.history = History()
        self.camera = camera
        self.snap = snap
        """Whether new and moved points jump to the nearest grid crossing."""
        self.selected: int | None = None
        self.cursor: Pixel = (camera.size[0] / 2, camera.size[1] / 2)
        """The mouse position in window pixels."""
        self.last_radius = DEFAULT_RADIUS
        """The radius the last corner was rounded to; the next one starts there."""
        self._gesture: _PointDrag | _Pan | _Rounding | None = None

    # ------------------------------------------------------------------ #
    # What's under the cursor
    # ------------------------------------------------------------------ #

    @property
    def cursor_world(self) -> Point:
        """The mouse position in world metres."""
        x, y = self.camera.to_world(self.cursor)
        return float(x), float(y)

    @property
    def hovered(self) -> int | None:
        """The point under the cursor, if any."""
        return self.draft.point_near(self.cursor_world, GRAB_RADIUS / self.camera.scale)

    @property
    def rounding(self) -> Rounding | None:
        """The corner being rounded, while C is held."""
        if not isinstance(self._gesture, _Rounding):
            return None
        return Rounding(self._gesture.index, self._gesture.radius, self._gesture.limits)

    @property
    def grid_step(self) -> float:
        """Grid spacing at the current zoom, in metres; snapping uses the same spacing."""
        return self.camera.grid_step()

    # ------------------------------------------------------------------ #
    # Mouse
    # ------------------------------------------------------------------ #

    def press(self, pixel: Pixel, button: Button, *, shift: bool = False) -> None:
        """A mouse button went down at ``pixel``."""
        self.cursor = pixel
        self._finish_gesture()  # a new press ends any gesture whose release never arrived
        hovered = self.hovered
        if button is Button.LEFT:
            before = self.draft
            if hovered is None:
                hovered = self._add_point(shift=shift)
            self.selected = hovered
            x, y = self.draft.points[hovered]
            cursor_x, cursor_y = self.cursor_world
            self._gesture = _PointDrag(before, hovered, (x - cursor_x, y - cursor_y), pixel)
        elif button is Button.RIGHT and hovered is not None:
            self._delete(hovered)
        else:
            self._gesture = _Pan(button, pixel)

    def move(self, pixel: Pixel) -> None:
        """The mouse moved to ``pixel``."""
        self.cursor = pixel
        match self._gesture:
            case _Pan(last=(last_x, last_y)) as pan:
                self.camera = self.camera.pan(pixel[0] - last_x, pixel[1] - last_y)
                pan.last = pixel
            case _PointDrag() as drag:
                if not drag.moving:
                    if np.hypot(*np.subtract(pixel, drag.pressed_at)) < DRAG_THRESHOLD:
                        return
                    drag.moving = True
                x, y = self.cursor_world
                target = self._snapped((x + drag.offset[0], y + drag.offset[1]))
                self.draft = self.draft.move_point(drag.index, target)

    def release(self, pixel: Pixel, button: Button) -> None:
        """A mouse button came up at ``pixel``."""
        self.move(pixel)
        match self._gesture:
            case _PointDrag() if button is Button.LEFT:
                self._finish_gesture()
            case _Pan(button=held) if button is held:
                self._finish_gesture()

    def scroll(self, notches: float, *, shift: bool = False) -> None:
        """The wheel turned ``notches`` (positive is away from the user) at the cursor.

        Zooms around the cursor; with shift, changes the road width at the point under the
        cursor, or at the selected point. While a corner is being rounded, changes its radius.
        """
        if isinstance(self._gesture, _Rounding):
            self._change_radius(self._gesture, notches)
            return
        if not shift:
            self.camera = self.camera.zoom_at(self.cursor, ZOOM_STEP**notches)
            return
        hovered = self.hovered
        target = hovered if hovered is not None else self.selected
        if target is not None:
            self.selected = target
            self.edit(
                self.draft.change_width(target, notches * WIDTH_STEP), merge=("width", target)
            )

    # ------------------------------------------------------------------ #
    # Editing and undo
    # ------------------------------------------------------------------ #

    def edit(self, draft: TrackDraft, *, merge: Hashable | None = None) -> None:
        """Make an edited draft the current one, as one undo step.

        Edits in a row with the same ``merge`` key make one step (see `History.record`). A
        gesture in progress (a drag, or rounding a corner) is finished first.
        """
        self._finish_gesture()
        self.history.record(self.draft, draft, merge=merge)
        self.draft = draft

    def undo(self) -> bool:
        """Take back the last edit; returns whether there was one.

        A drag still in progress counts as finished first, so undoing it puts the point back.
        """
        self._finish_gesture()
        return self._go_to(self.history.undo(self.draft))

    def redo(self) -> bool:
        """Put back the last edit undone; returns whether there was one."""
        self._finish_gesture()
        return self._go_to(self.history.redo(self.draft))

    # ------------------------------------------------------------------ #
    # Rounding a corner (hold C, turn the wheel, let go)
    # ------------------------------------------------------------------ #

    def start_rounding(self) -> None:
        """Round the corner under the cursor (or the selected one) at the radius used last.

        Does nothing if a corner is already being rounded (a held key repeats).

        Raises:
            mlracecar.editor.corners.CornerError: If there's no point to round, it isn't a
                corner, or no radius fits there. The message says which.
        """
        if isinstance(self._gesture, _Rounding):
            return
        self._finish_gesture()
        hovered = self.hovered
        index = hovered if hovered is not None else self.selected
        if index is None:
            raise CornerError("point at a corner, or select one, to round it")
        limits = self.draft.corner_limits(index)
        rounding = _Rounding(self.draft, index, limits, self.last_radius)
        self._gesture = rounding
        self.draft, self.selected = rounding.before.round_corner(index, rounding.radius)

    def finish_rounding(self) -> None:
        """Keep the rounded corner (C was let go), as one undo step."""
        if isinstance(self._gesture, _Rounding):
            self._finish_gesture()

    def cancel_rounding(self) -> bool:
        """Put the corner back as it was; returns whether a corner was being rounded."""
        if not isinstance(self._gesture, _Rounding):
            return False
        self.draft, self.selected = self._gesture.before, self._gesture.index
        self._gesture = None
        return True

    # ------------------------------------------------------------------ #
    # Keyboard actions
    # ------------------------------------------------------------------ #

    def pan(self, dx: float, dy: float) -> None:
        """Move the view by ``(dx, dy)`` pixels, as if dragged."""
        self.camera = self.camera.pan(dx, dy)

    def zoom(self, notches: float) -> None:
        """Zoom around the middle of the window (positive zooms in)."""
        width, height = self.camera.size
        self.camera = self.camera.zoom_at((width / 2, height / 2), ZOOM_STEP**notches)

    def fit(self) -> None:
        """Zoom to show the whole track (or every point, while it isn't a track yet)."""
        track = self.draft.track
        points = self.draft.points if track is None else np.concatenate([track.left, track.right])
        self.camera = self.camera.fit(points)

    def toggle_snap(self) -> None:
        """Turn snapping to the grid on or off."""
        self.snap = not self.snap

    def delete_selected(self) -> None:
        """Delete the selected point."""
        if self.selected is not None:
            self._delete(self.selected)

    def change_selected_width(self, by: float) -> None:
        """Widen (positive) or narrow the road at the selected point, in metres."""
        if self.selected is not None:
            self.edit(self.draft.change_width(self.selected, by), merge=("width", self.selected))

    def reverse(self) -> None:
        """Drive the track the other way round. The selection stays on the same point."""
        count = len(self.draft.points)
        self.edit(self.draft.reverse())
        if self.selected is not None:
            self.selected = (count - self.selected) % count

    def start_at_selected(self) -> None:
        """Move the start/finish line to the selected point."""
        if self.selected is not None:
            self.edit(self.draft.set_start(self.selected))
            self.selected = 0

    def deselect(self) -> None:
        """Clear the selection."""
        self.selected = None

    def resize(self, size: tuple[int, int]) -> None:
        """The window changed size."""
        self.camera = self.camera.resized(size)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _finish_gesture(self) -> None:
        """End the gesture in progress. A drag (with the point its press may have added) or a
        rounding is one undo step."""
        match self._gesture:
            case _PointDrag(before=before):
                self.history.record(before, self.draft)
            case _Rounding(before=before) as rounding:
                self.history.record(before, self.draft)
                self.last_radius = rounding.radius
        self._gesture = None

    def _change_radius(self, rounding: _Rounding, notches: float) -> None:
        limits = rounding.limits
        old = rounding.radius
        wanted = rounding.wanted * RADIUS_STEP**notches
        rounding.wanted = float(np.clip(wanted, limits.smallest, limits.largest))
        if rounding.radius != old:
            self.draft, self.selected = rounding.before.round_corner(
                rounding.index, rounding.radius
            )

    def _go_to(self, draft: TrackDraft | None) -> bool:
        """Switch to a draft from the history, if there is one."""
        if draft is None:
            return False
        self.draft = draft
        if self.selected is not None and self.selected >= len(draft.points):
            self.selected = None
        return True

    def _snapped(self, position: Point) -> Point:
        if not self.snap:
            return position
        step = self.grid_step
        return round(position[0] / step) * step, round(position[1] / step) * step

    def _add_point(self, *, shift: bool) -> int:
        """Add a point at the cursor and return its index."""
        position = self._snapped(self.cursor_world)
        if shift or not self.draft.is_on_road(self.cursor_world):
            self.draft = self.draft.append_point(position)
            return len(self.draft.points) - 1
        before = self.draft.points
        self.draft = self.draft.insert_point(position)
        # The new point is where the lists first differ (or at the end, if it was appended).
        pairs = zip(before, self.draft.points, strict=False)
        return next((i for i, (old, new) in enumerate(pairs) if old != new), len(before))

    def _delete(self, index: int) -> None:
        self.edit(self.draft.delete_point(index))
        if self.selected == index:
            self.selected = None
        elif self.selected is not None and self.selected > index:
            self.selected -= 1
