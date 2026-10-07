"""The editor window: a pygame loop that feeds input to the controller and draws the view.

This is the only part of the editor that deals with pygame events. It also owns what sits
around the editing: the file (`TrackDocument`), the background track checks, dialogs, and short
messages. Keyboard shortcuts are listed once, in `SHORTCUTS`, which also fills the help panel,
so the help can't drift from the keys.
"""

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pygame

from mlracecar.core.track.validation import Severity
from mlracecar.editor.checks import BackgroundChecks
from mlracecar.editor.controller import PAN_STEP, WIDTH_STEP, Button, EditorController
from mlracecar.editor.document import TrackDocument
from mlracecar.editor.draft import TrackDraft
from mlracecar.editor.view import (
    ERROR,
    READY,
    WARNING,
    DialogBox,
    EditorView,
    Message,
    Overlays,
)
from mlracecar.io.track_file import TrackFileError
from mlracecar.render.camera import Camera

WINDOW_SIZE = (1280, 800)
"""The window's size, unless the screen is too small for it."""

SCREEN_SHARE = 0.85
"""At most this share of the screen's width and height goes to the window."""

FRAME_RATE = 60

MESSAGE_SECONDS = 4.0
"""How long a message (like "Saved ...") stays on screen."""

ENTER_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER)


@dataclass
class Dialog:
    """A question over the editor. While it's open, it gets all keyboard input; Esc cancels."""

    title: str
    hint: str
    answers: Mapping[int, Callable[[], None]] = field(default_factory=dict)
    """Keys that answer it, and what each one does."""
    text: str | None = None
    """What's been typed so far, for questions answered with text."""
    submit: Callable[[str], None] | None = None
    """Called with the text when Enter is pressed, for questions answered with text."""


class EditorWindow:
    """The editor in a pygame window. Create it, then `run` it until the window is closed."""

    def __init__(
        self,
        document: TrackDocument,
        size: tuple[int, int] | None = None,
        checks: BackgroundChecks | None = None,
    ) -> None:
        """Open the window.

        Args:
            document: The track to edit.
            size: Window size in pixels; by default `WINDOW_SIZE`, shrunk to fit the screen.
            checks: Runs the track checks; by default on a worker thread.
        """
        pygame.display.init()
        pygame.font.init()
        size = size or _fitting_window_size()
        self._screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.key.set_repeat(300, 30)  # holding a key repeats it, e.g. to keep panning
        self.document = document
        self.controller = EditorController(document.saved, Camera(size=size))
        if document.saved.points:
            self.controller.fit()
        self.checks = checks or BackgroundChecks(
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="track-checks")
        )
        self.show_help = not document.saved.points
        """Whether the help panel is showing. It starts open for a new, empty track."""
        self.show_issue_list = True
        self.dialog: Dialog | None = None
        self.message: Message | None = None
        self.running = True
        """Becomes false when the window closes."""
        self._message_until = 0
        self._caption = ""
        self._view = EditorView()

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def handle(self, event: pygame.event.Event) -> None:
        """Apply one pygame event."""
        if event.type == pygame.QUIT:
            self.close()
        elif self.dialog is not None:
            self._answer(self.dialog, event)
        else:
            self._edit(event)

    def _edit(self, event: pygame.event.Event) -> None:
        editor = self.controller
        shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
        match event.type:
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
            case pygame.KEYDOWN:
                ctrl = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))  # Cmd on a Mac
                shifted = bool(event.mod & pygame.KMOD_SHIFT)
                action = _KEY_ACTIONS.get(Chord(event.key, ctrl, shifted)) or _KEY_ACTIONS.get(
                    Chord(event.key, ctrl)  # e.g. "+" is shift+= on many keyboards
                )
                if action is not None:
                    action(self)

    def _answer(self, dialog: Dialog, event: pygame.event.Event) -> None:
        match event.type:
            case pygame.TEXTINPUT if dialog.text is not None:
                dialog.text += event.text
            case pygame.KEYDOWN if event.key == pygame.K_ESCAPE:
                self.dialog = None
            case pygame.KEYDOWN if event.key == pygame.K_BACKSPACE and dialog.text is not None:
                dialog.text = dialog.text[:-1]
            case pygame.KEYDOWN if event.key in ENTER_KEYS and dialog.submit is not None:
                self.dialog = None
                dialog.submit(dialog.text or "")
            case pygame.KEYDOWN if event.key in dialog.answers:
                self.dialog = None
                dialog.answers[event.key]()

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def save(self, then: Callable[[], None] | None = None) -> None:
        """Save to the track's file, asking for a file name first if it has none.

        ``then`` runs once the track is saved (not if saving is cancelled or fails).
        """
        if self.document.path is None:
            self.save_as(then)
        else:
            self._write(self.document.path, then)

    def save_as(self, then: Callable[[], None] | None = None) -> None:
        """Ask for a file name, then save there."""
        suggestion = self.document.suggested_path(self.controller.draft)
        self.dialog = Dialog(
            "Save the track as:",
            "Enter: save   Esc: cancel",
            text=suggestion.as_posix(),  # forward slashes work everywhere, Windows included
            submit=lambda text: self._save_to(text, then),
        )

    def rename(self) -> None:
        """Ask for a new name for the track."""
        self.dialog = Dialog(
            "Track name:",
            "Enter: rename   Esc: cancel",
            text=self.controller.draft.name,
            submit=self._rename_to,
        )

    def close(self) -> None:
        """Close the window, asking first if there are unsaved changes."""
        if not self.document.is_modified(self.controller.draft):
            self.running = False
            return

        def save_and_close() -> None:
            self.save(then=self._stop)

        self.dialog = Dialog(
            f'Save the changes to "{self.controller.draft.name}" before closing?',
            "Enter: save   Q: close without saving   Esc: keep editing",
            answers={**dict.fromkeys(ENTER_KEYS, save_and_close), pygame.K_q: self._stop},
        )

    def undo(self) -> None:
        """Take back the last edit."""
        if not self.controller.undo():
            self._show("Nothing to undo.", WARNING)

    def redo(self) -> None:
        """Put back the last edit undone."""
        if not self.controller.redo():
            self._show("Nothing to redo.", WARNING)

    def toggle_help(self) -> None:
        """Show or hide the help panel."""
        self.show_help = not self.show_help

    def toggle_issue_list(self) -> None:
        """Show or hide the list of track problems."""
        self.show_issue_list = not self.show_issue_list

    def _save_to(self, text: str, then: Callable[[], None] | None) -> None:
        if not text.strip():
            self._show("Type a file name to save to.", WARNING)
            return
        path = Path(text.strip())
        if path.suffix != ".json":
            path = path.with_name(path.name + ".json")
        if path.exists() and path != self.document.path:
            self.dialog = Dialog(
                f"{path} already exists. Replace it?",
                "Enter: replace   Esc: cancel",
                answers=dict.fromkeys(ENTER_KEYS, lambda: self._write(path, then)),
            )
            return
        self._write(path, then)

    def _write(self, path: Path, then: Callable[[], None] | None) -> None:
        try:
            saved_to = self.document.save(self.controller.draft, path)
        except (TrackFileError, OSError) as error:
            self._show(f"Can't save: {error}", ERROR)
            return
        # Saving may name the track after the file and round its numbers. That's part of the
        # save, not an edit to undo.
        self.controller.draft = self.document.saved
        errors = sum(issue.severity is Severity.ERROR for issue in self.document.saved.issues)
        if errors:
            plural = "s" if errors > 1 else ""
            self._show(
                f"Saved {saved_to}, but with {errors} error{plural}: can't race yet.", WARNING
            )
        else:
            self._show(f"Saved {saved_to}.", READY)
        if then is not None:
            then()

    def _rename_to(self, name: str) -> None:
        self.controller.edit(self.controller.draft.rename(name))

    def _stop(self) -> None:
        self.running = False

    def _show(self, text: str, color: tuple[int, int, int]) -> None:
        self.message = Message(text, color)
        self._message_until = pygame.time.get_ticks() + round(MESSAGE_SECONDS * 1000)

    # ------------------------------------------------------------------ #
    # Drawing and the main loop
    # ------------------------------------------------------------------ #

    def overlays(self) -> Overlays:
        """What to show on top of the editor right now."""
        result = self.checks.result
        # Results for an older draft still line up while the points are the same ones (during
        # a drag, say); after an insert or delete, their point numbers would be off.
        usable = result is not None and len(result.draft.points) == len(
            self.controller.draft.points
        )
        dialog = self.dialog
        return Overlays(
            issues=result.issues if result is not None and usable else (),
            checking=not usable,
            show_issue_list=self.show_issue_list,
            help_lines=HELP_LINES if self.show_help else (),
            message=self.message,
            dialog=None if dialog is None else DialogBox(dialog.title, dialog.hint, dialog.text),
        )

    def draw(self) -> None:
        """Draw the editor and show it."""
        self._update_caption()
        self._view.draw(self._screen, self.controller, self.overlays())
        pygame.display.flip()

    def run(self) -> TrackDraft:
        """Run until the window is closed, and return the final draft."""
        clock = pygame.time.Clock()
        changed = True  # draw the first frame straight away
        while self.running:
            if changed:
                self.draw()  # only redraw when something changed: an idle editor uses no CPU
            events = pygame.event.get()
            for event in events:
                self.handle(event)
            self.checks.check(self.controller.draft)
            # `|`, not `or`: the poll and the message timer must both run every frame.
            changed = bool(events) | self.checks.poll() | self._expire_message()
            clock.tick(FRAME_RATE)
        self.checks.shutdown()
        pygame.display.quit()
        return self.controller.draft

    def _expire_message(self) -> bool:
        if self.message is None or pygame.time.get_ticks() < self._message_until:
            return False
        self.message = None
        return True

    def _update_caption(self) -> None:
        draft, path = self.controller.draft, self.document.path
        caption = draft.name + (f" ({path.name})" if path else "")
        if self.document.is_modified(draft):
            caption += " *"
        caption += " - Track editor"
        if caption != self._caption:
            pygame.display.set_caption(caption)
            self._caption = caption


class Chord(NamedTuple):
    """A key, plus whether Ctrl (or Cmd) and Shift are held."""

    key: int
    ctrl: bool = False
    shift: bool = False


type Action = Callable[[EditorWindow], None]


class Shortcut(NamedTuple):
    """Keys that do one thing (or one thing and its opposite), as the help panel lists them."""

    label: str
    description: str
    actions: Mapping[Chord, Action]


def _on_editor(action: Callable[[EditorController], None]) -> Action:
    """A shortcut action that only needs the editor's controller."""
    return lambda window: action(window.controller)


SHORTCUTS = (
    Shortcut(
        "Ctrl+Z / Ctrl+Y",
        "undo / redo (Ctrl+Shift+Z also redoes)",
        {
            Chord(pygame.K_z, ctrl=True): EditorWindow.undo,
            Chord(pygame.K_y, ctrl=True): EditorWindow.redo,
            Chord(pygame.K_z, ctrl=True, shift=True): EditorWindow.redo,
        },
    ),
    Shortcut("Ctrl+S", "save", {Chord(pygame.K_s, ctrl=True): EditorWindow.save}),
    Shortcut(
        "Ctrl+Shift+S",
        "save as a new file",
        {Chord(pygame.K_s, ctrl=True, shift=True): EditorWindow.save_as},
    ),
    Shortcut("F2", "rename the track", {Chord(pygame.K_F2): EditorWindow.rename}),
    Shortcut(
        "Arrow keys",
        "pan",
        {
            Chord(pygame.K_LEFT): _on_editor(lambda editor: editor.pan(PAN_STEP, 0)),
            Chord(pygame.K_RIGHT): _on_editor(lambda editor: editor.pan(-PAN_STEP, 0)),
            Chord(pygame.K_UP): _on_editor(lambda editor: editor.pan(0, PAN_STEP)),
            Chord(pygame.K_DOWN): _on_editor(lambda editor: editor.pan(0, -PAN_STEP)),
        },
    ),
    Shortcut(
        "+ / -",
        "zoom in / out",
        {
            **dict.fromkeys(
                [Chord(pygame.K_EQUALS), Chord(pygame.K_PLUS), Chord(pygame.K_KP_PLUS)],
                _on_editor(lambda editor: editor.zoom(1)),
            ),
            **dict.fromkeys(
                [Chord(pygame.K_MINUS), Chord(pygame.K_KP_MINUS)],
                _on_editor(lambda editor: editor.zoom(-1)),
            ),
        },
    ),
    Shortcut(
        "F", "fit the track in the window", {Chord(pygame.K_f): _on_editor(EditorController.fit)}
    ),
    Shortcut(
        "G",
        "snap to the grid on / off",
        {Chord(pygame.K_g): _on_editor(EditorController.toggle_snap)},
    ),
    Shortcut(
        "[ / ]",
        "narrow / widen the road at the selected point",
        {
            Chord(pygame.K_LEFTBRACKET): _on_editor(
                lambda editor: editor.change_selected_width(-WIDTH_STEP)
            ),
            Chord(pygame.K_RIGHTBRACKET): _on_editor(
                lambda editor: editor.change_selected_width(WIDTH_STEP)
            ),
        },
    ),
    Shortcut(
        "Delete",
        "delete the selected point",
        dict.fromkeys(
            [Chord(pygame.K_DELETE), Chord(pygame.K_BACKSPACE)],
            _on_editor(EditorController.delete_selected),
        ),
    ),
    Shortcut(
        "S",
        "put the start/finish line at the selected point",
        {Chord(pygame.K_s): _on_editor(EditorController.start_at_selected)},
    ),
    Shortcut(
        "R",
        "reverse the driving direction",
        {Chord(pygame.K_r): _on_editor(EditorController.reverse)},
    ),
    Shortcut(
        "Esc",
        "clear the selection",
        {Chord(pygame.K_ESCAPE): _on_editor(EditorController.deselect)},
    ),
    Shortcut(
        "I", "show / hide the list of problems", {Chord(pygame.K_i): EditorWindow.toggle_issue_list}
    ),
    Shortcut("H", "show / hide this help", {Chord(pygame.K_h): EditorWindow.toggle_help}),
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

HELP_LINES = (
    *MOUSE_HELP,
    *((shortcut.label, shortcut.description) for shortcut in SHORTCUTS),
)

_KEY_ACTIONS = {
    chord: action for shortcut in SHORTCUTS for chord, action in shortcut.actions.items()
}


def _fitting_window_size() -> tuple[int, int]:
    """`WINDOW_SIZE`, shrunk to fit on smaller screens."""
    screen_width, screen_height = pygame.display.get_desktop_sizes()[0]
    return (
        min(WINDOW_SIZE[0], int(screen_width * SCREEN_SHARE)),
        min(WINDOW_SIZE[1], int(screen_height * SCREEN_SHARE)),
    )


def run_editor(document: TrackDocument | None = None) -> TrackDraft:
    """Open the editor on ``document`` (a new, empty track if none) and return the final draft."""
    return EditorWindow(document if document is not None else TrackDocument.open()).run()
