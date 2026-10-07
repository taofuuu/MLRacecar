"""Write configs/default.yaml: every setting with its default value and a short explanation.

Run after adding or changing a setting:

    uv run python scripts/export_default_config.py

A test fails while the file is out of date, so it can't silently drift from the code.
"""

from pathlib import Path

from mlracecar.config.files import format_config
from mlracecar.config.models import RacecarConfig

DEFAULT_CONFIG_PATH = Path(__file__).parents[1] / "configs" / "default.yaml"

if __name__ == "__main__":
    DEFAULT_CONFIG_PATH.parent.mkdir(exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(format_config(RacecarConfig()), encoding="utf-8", newline="\n")
    print(f"wrote {DEFAULT_CONFIG_PATH}")
