"""Print the simulation speed table for the README from a pytest-benchmark results file.

    uv run pytest -m benchmark --no-cov --benchmark-json=benchmark.json
    uv run python scripts/benchmark_table.py benchmark.json

The table shows the world step measurements (`tests/benchmarks/test_world_speed.py`): how long
one step takes for each number of cars, and how many steps, car-steps, and seconds of racing
that makes per second. CI adds the same table to the summary of every benchmark run.
"""

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WORLD_STEP = "test_world_step["
HEADER = (
    "| Cars | Time per step | Steps per second | Car-steps per second | Faster than real time |\n"
    "|-----:|--------------:|-----------------:|---------------------:|----------------------:|"
)


def speed_table(results: Mapping[str, Any]) -> str:
    """The README table for the world step results, then a line saying where they were measured.

    Raises:
        ValueError: If the results have no world step measurements.
    """
    steps = [bench for bench in results["benchmarks"] if bench["name"].startswith(WORLD_STEP)]
    if not steps:
        raise ValueError("no world step results: run tests/benchmarks/test_world_speed.py")
    rows = [HEADER]
    for bench in sorted(steps, key=lambda bench: bench["extra_info"]["cars"]):
        cars = bench["extra_info"]["cars"]
        seconds = bench["stats"]["median"]
        rows.append(
            f"| {cars:,} | {seconds * 1e3:.2f} ms | {1 / seconds:,.0f} | {cars / seconds:,.0f} "
            f"| {bench['extra_info']['decision_dt'] / seconds:,.0f}x |"
        )
    machine = results["machine_info"]
    cpu = machine["cpu"].get("brand_raw") or machine["processor"]
    rounds = steps[0]["stats"]["rounds"]
    rows.append(
        f"\nMedian of {rounds} steps on {cpu} ({machine['system']} {machine['release']}), "
        f"Python {machine['python_version']}, NumPy {machine.get('numpy', 'unknown')}, "
        f"commit {results['commit_info'].get('id', 'unknown')[:7]}."
    )
    return "\n".join(rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python scripts/benchmark_table.py RESULTS.json")
    try:
        print(speed_table(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
    except ValueError as error:
        sys.exit(f"{sys.argv[1]}: {error}")
