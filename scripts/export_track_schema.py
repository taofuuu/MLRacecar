"""Write the track file JSON Schema to docs/schemas/, where the docs site publishes it.

Run after changing the track file format:

    uv run python scripts/export_track_schema.py

A test fails while the published schema is out of date, so it can't silently drift.
"""

from pathlib import Path

from mlracecar.io.track_file import CURRENT_VERSION, track_file_schema

SCHEMA_PATH = Path(__file__).parents[1] / "docs" / "schemas" / f"track-file-v{CURRENT_VERSION}.json"

if __name__ == "__main__":
    SCHEMA_PATH.write_text(track_file_schema(), encoding="utf-8", newline="\n")
    print(f"wrote {SCHEMA_PATH}")
