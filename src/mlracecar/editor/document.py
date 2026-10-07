"""The file behind the editor: where the track is saved, and whether it has unsaved changes.

Drafts are immutable (ADR-0011), so "unsaved changes" is a comparison: the draft on screen
against the draft last opened or saved. No pygame here; the window decides when to save.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from mlracecar.editor.draft import DEFAULT_NAME, TrackDraft
from mlracecar.io.track_file import read_track_file, write_track_file

TRACK_FOLDER = Path("tracks")
"""Where new tracks are suggested to go, if that folder exists."""

SAVED_DECIMALS = 3
"""Positions and widths are saved to this many decimal places: the nearest millimetre, so files
stay readable (``-314.084``, not ``-314.0837535325377``). Not coarser: a rounded corner's points
can be a metre or two apart, and centimetre steps that close together bend the curve visibly."""


@dataclass
class TrackDocument:
    """A track file being edited: its path (``None`` until first saved) and its saved state."""

    path: Path | None
    saved: TrackDraft
    """The draft as it is on disk (or as it started, for a new track)."""

    @classmethod
    def open(cls, path: Path | None = None) -> Self:
        """Open a track file, or start a new track: unnamed with no path, or named after a path
        that doesn't exist yet.

        Raises:
            mlracecar.io.track_file.TrackFileError: If the file exists but can't be read.
        """
        if path is None:
            return cls(None, TrackDraft())
        if path.exists():
            return cls(path, TrackDraft.from_track_file(read_track_file(path)))
        return cls(path, TrackDraft(name=path.stem))

    def is_modified(self, draft: TrackDraft) -> bool:
        """Whether ``draft`` has changes that aren't saved."""
        return draft != self.saved

    def save(self, draft: TrackDraft, path: Path | None = None) -> Path:
        """Save ``draft`` to ``path`` (by default the document's own path) and return where.

        Positions and widths are rounded to `SAVED_DECIMALS`, and a track still called
        `DEFAULT_NAME` is named after the file, so afterwards `saved` (not ``draft``) is what's
        on disk. Missing folders are created. Saving works with track errors, since a draft is
        work in progress.

        Raises:
            ValueError: If there's nowhere to save (no ``path`` and the document has none).
            mlracecar.io.track_file.TrackFileError: If the draft can't be a track file yet
                (fewer than 3 points).
            OSError: If the file can't be written.
        """
        target = path or self.path
        if target is None:
            raise ValueError("the track has no file yet; choose where to save it")
        if draft.name == DEFAULT_NAME:
            draft = draft.rename(target.stem)
        draft = draft.rounded(SAVED_DECIMALS)
        track_file = draft.to_track_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_track_file(track_file, target)
        self.path, self.saved = target, draft
        return target

    def suggested_path(self, draft: TrackDraft) -> Path:
        """Where to offer to save: the current file, or a name made from the track's name."""
        if self.path is not None:
            return self.path
        stem = re.sub(r"[^a-z0-9]+", "-", draft.name.lower()).strip("-") or "track"
        folder = TRACK_FOLDER if TRACK_FOLDER.is_dir() else Path()
        return folder / f"{stem}.json"
