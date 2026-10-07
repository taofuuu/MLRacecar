"""Record the golden trajectories again, after an intended change to the physics or race rules.

    uv run python scripts/update_golden.py

The regression tests compare every run with these files, so record them again only when a
change is meant, and say in the pull request what changed and why.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tests"))  # the scenarios and the drivers
from golden import SCENARIOS, write

if __name__ == "__main__":
    for scenario in SCENARIOS:
        print(f"wrote {write(scenario)}")
    print("Now say in the pull request what changed and why the golden files changed with it.")
