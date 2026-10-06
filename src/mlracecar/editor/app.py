"""The editor window: a pygame loop that feeds input to the controller and draws the view.

This is the only part of the editor that deals with pygame events. Keyboard shortcuts are listed
once, in `SHORTCUTS`, which also fills the help panel, so the help can't drift from the keys.
"""

from collections.abc import Callable, Mapping
from typing import NamedTuple

import pygame

from mlracecar.editor.controller import PAN_STEP, WIDTH_STEP, Button, EditorController
from mlracecar.editor.draft import TrackDraft
from mlracecar.editor.view import EditorView
from mlracecar.render.camera import Camera

WINDOW_SIZE = (1280, 800)
"""The window's size, unless the screen is too small for it."""

SCREEN_SHARE = 0.85
"""At most this share of the screen's width and height goes to the window."""

FRAME_RATE = 60

type KeyAction = Callable[[EditorController], None]


class Shortcut(NamedTuple):
    """Keys that do one thing (or one thing and its opposite), as the help panel lists them."""

    label: str
    description: str
    actions: Mapping[int, KeyAction]
    """pygame key code -> what that key does."""


def _zoom_in(editor: EditorController) -> None:
    editor.zoom(1)


def _zoom_out(editor: EditorController) -> None:
    editor.zoom(-1)


SHORTCUTS = (
    Shortcut(
        "Arrow keys",
        "pan",
        {
            pygame.K_LEFT: lambda editor: editor.pan(PAN_STEP, 0),
            pygame.K_RIGHT: lambda editor: editor.pan(-PAN_STEP, 0),
            pygame.K_UP: lambda editor: editor.pan(0, PAN_STEP),
            pygame.K_DOWN: lambda editor: editor.pan(0, -PAN_STEP),
        },
    ),
    Shortcut(
        "+ / -",
        "zoom in / out",
        {
            pygame.K_EQUALS: _zoom_in,
            pygame.K_PLUS: _zoom_in,
            pygame.K_KP_PLUS: _zoom_in,
            pygame.K_MINUS: _zoom_out,
            pygame.K_KP_MINUS: _zoom_out,
        },
    ),
    Shortcut("F", "fit the track in the window", {pygame.K_f: EditorController.fit}),
    Shortcut("G", "snap to the grid on / off", {pygame.K_g: EditorController.toggle_snap}),
    Shortcut(
        "[ / ]",
        "narrow / widen the road at the selected point",
        {
            pygame.K_LEFTBRACKET: lambda editor: editor.change_selected_width(-WIDTH_STEP),
            pygame.K_RIGHTBRACKET: lambda editor: editor.change_selected_width(WIDTH_STEP),
        },
    ),
    Shortcut(
        "Delete",
        "delete the selected point",
        {
            pygame.K_DELETE: EditorController.delete_selected,
            pygame.K_BACKSPACE: EditorController.delete_selected,
        },
    ),
    Shortcut(
        "S",
        "put the start/finish line at the selected point",
        {pygame.K_s: EditorController.start_at_selected},
    ),
    Shortcut("R", "reverse the driving direction", {pygame.K_r: EditorController.reverse}),
    Shortcut("Esc", "clear the selection", {pygame.K_ESCAPE: EditorController.deselect}),
)

MOUSE_HELP = (
    ("Click on the road", "insert a point there (hold to place it)"),
    ("Click elsewhere", "add a point after the last one"),
    ("Shift+click", "always add after the last one"),
    ("Drag a point", "move it"),
    ("Right-click a point", "delete it"),
    ("Right-drag", "pan"),
    ("Wheel", "zoom"),
    ("Shift+wheel", "narrow / widen the road at a point"),
)

HELP_KEY = pygame.K_h

HELP_LINES = (
    *MOUSE_HELP,
    *((shortcut.label, shortcut.description) for shortcut in SHORTCUTS),
    ("H", "show / hide this help"),
)

_KEY_ACTIONS = {key: action for shortcut in SHORTCUTS for key, action in shortcut.actions.items()}


class EditorWindow:
    """The editor in a pygame window. Create it, then `run` it until the window is closed."""

    def __init__(self, draft: TrackDraft, size: tuple[int, int] | None = None) -> None:
        pygame.display.init()
        pygame.font.init()
        size = size or _fitting_window_size()
        self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption(f"Track editor: {draft.name}")
        pygame.key.set_repeat(300, 30)  # holding a key repeats it, e.g. to keep panning
        self.controller = EditorController(draft, Camera(size=size))
        if draft.points:
            self.controller.fit()
        self.show_help = not draft.points
        """Whether the help panel is showing. It starts open for a new, empty track."""
        self.running = True
        """Becomes false when the window is closed."""
        self._view = EditorView()

    def handle(self, event: pygame.event.Event) -> None:
        """Apply one pygame event."""
        editor = self.controller
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        match event.type:
            case pygame.QUIT:
                self.running = False
            case pygame.VIDEORESIZE:
                editor.resize(event.size)
            case pygame.MOUSEMOTION:
                editor.move(event.pos)
            case pygame.MOUSEBUTTONDOWN if event.button in Button:
                editor.press(event.pos, Button(event.button), shift=shift)
            case pygame.MOUSEBUTTONUP if event.button in Button:
                editor.release(event.pos, Button(event.button))
            case pygame.MOUSEWHEEL if shift:
                editor.scroll(event.y or event.x, shift=True)  # some systems send shift+wheel as x
            case pygame.MOUSEWHEEL:
                editor.scroll(event.precise_y)
            case pygame.KEYDOWN if event.key == HELP_KEY:
                self.show_help = not self.show_help
            case pygame.KEYDOWN if event.key in _KEY_ACTIONS:
                _KEY_ACTIONS[event.key](editor)

    def draw(self) -> None:
        """Draw the editor and show it."""
        self._view.draw(self._screen, self.controller, HELP_LINES if self.show_help else ())
        pygame.display.flip()

    def run(self) -> TrackDraft:
        """Run until the window is closed, and return the final draft."""
        clock = pygame.time.Clock()
        changed = True  # draw the first frame straight away
        while self.running:
            if changed:
                self.draw()  # only redraw after input: an idle editor uses no CPU
            events = pygame.event.get()
            for event in events:
                self.handle(event)
            changed = bool(events)
            clock.tick(FRAME_RATE)
        pygame.display.quit()
        return self.controller.draft


def _fitting_window_size() -> tuple[int, int]:
    """`WINDOW_SIZE`, shrunk to fit on smaller screens."""
    screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
    return (
        min(WINDOW_SIZE[0], int(screen_width * SCREEN_SHARE)),
        min(WINDOW_SIZE[1], int(screen_height * SCREEN_SHARE)),
    )


def run_editor(draft: TrackDraft | None = None) -> TrackDraft:
    """Open the editor on ``draft`` (a new, empty track if none) and return the final draft."""
    return EditorWindow(draft if draft is not None else TrackDraft()).run()
