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
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from mlracecar.editor.draft import Point, TrackDraft
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


class Button(IntEnum):
    """Mouse buttons, numbered as pygame and SDL number them."""

    LEFT = 1
    MIDDLE = 2
    RIGHT = 3


@dataclass
class _PointDrag:
    index: int
    offset: Point
    """From the cursor to the point at the press, in metres, so the point doesn't jump."""
    pressed_at: Pixel
    moving: bool = False


@dataclass
class _Pan:
    button: Button
    last: Pixel


class EditorController:
    """The editor's state: the draft, the camera, and what the user is pointing at or holding.

    The draft itself is immutable (ADR-0011); the controller swaps in a new one on every edit.
    """

    def __init__(self, draft: TrackDraft, camera: Camera, *, snap: bool = False) -> None:
        self.draft = draft
        self.camera = camera
        self.snap = snap
        """Whether new and moved points jump to the nearest grid crossing."""
        self.selected: int | None = None
        self.cursor: Pixel = (camera.size[0] / 2, camera.size[1] / 2)
        """The mouse position in window pixels."""
        self._gesture: _PointDrag | _Pan | None = None

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
    def grid_step(self) -> float:
        """Grid spacing at the current zoom, in metres; snapping uses the same spacing."""
        return self.camera.grid_step()

    # ------------------------------------------------------------------ #
    # Mouse
    # ------------------------------------------------------------------ #

    def press(self, pixel: Pixel, button: Button, *, shift: bool = False) -> None:
        """A mouse button went down at ``pixel``."""
        self.cursor = pixel
        self._gesture = None  # a new press ends any gesture whose release never arrived
        hovered = self.hovered
        if button is Button.LEFT:
            if hovered is None:
                hovered = self._add_point(shift=shift)
            self.selected = hovered
            x, y = self.draft.points[hovered]
            cursor_x, cursor_y = self.cursor_world
            self._gesture = _PointDrag(hovered, (x - cursor_x, y - cursor_y), pixel)
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
                self._gesture = None
            case _Pan(button=held) if button is held:
                self._gesture = None

    def scroll(self, notches: float, *, shift: bool = False) -> None:
        """The wheel turned ``notches`` (positive is away from the user) at the cursor.

        Zooms around the cursor; with shift, changes the road width at the point under the
        cursor, or at the selected point.
        """
        if not shift:
            self.camera = self.camera.zoom_at(self.cursor, ZOOM_STEP**notches)
            return
        hovered = self.hovered
        target = hovered if hovered is not None else self.selected
        if target is not None:
            self.selected = target
            self.draft = self.draft.change_width(target, notches * WIDTH_STEP)

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
            self.draft = self.draft.change_width(self.selected, by)

    def reverse(self) -> None:
        """Drive the track the other way round. The selection stays on the same point."""
        count = len(self.draft.points)
        self.draft = self.draft.reverse()
        if self.selected is not None:
            self.selected = (count - self.selected) % count

    def start_at_selected(self) -> None:
        """Move the start/finish line to the selected point."""
        if self.selected is not None:
            self.draft = self.draft.set_start(self.selected)
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
        self.draft = self.draft.delete_point(index)
        if self.selected == index:
            self.selected = None
        elif self.selected is not None and self.selected > index:
            self.selected -= 1
