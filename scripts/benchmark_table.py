"""Print the simulation speed tables for the README from a pytest-benchmark results file.

    uv run pytest -m benchmark --no-cov --benchmark-json=benchmark.json
    uv run python scripts/benchmark_table.py benchmark.json

The first table shows the world step measurements (`tests/benchmarks/test_world_speed.py`): how
long one step takes for each number of cars, and how many steps, car-steps, and seconds of
racing that makes per second. The next ones show the distance sensors
(`tests/benchmarks/test_sensor_speed.py`) and the RL environment
(`tests/benchmarks/test_env_speed.py`), when they were measured too. CI adds the same tables to
the summary of every benchmark run.
"""

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WORLD_STEP = "test_world_step["
SENSE = "test_sense["
ONE_WORLD = "test_one_world["
SEPARATE = "test_separate_environments["
WORLD_HEADER = (
    "| Cars | Time per step | Steps per second | Car-steps per second | Faster than real time |\n"
    "|-----:|--------------:|-----------------:|---------------------:|----------------------:|"
)
SENSE_HEADER = (
    "| Cars | Time to read every ray | Car readings per second |\n"
    "|-----:|-----------------------:|------------------------:|"
)
ENV_HEADER = (
    "| Environment | Cars | Time per step | Car-steps per second |\n"
    "|-------------|-----:|--------------:|---------------------:|"
)


def speed_table(results: Mapping[str, Any]) -> str:
    """The README tables for the world step, sensor, and environment results, then where they
    were measured.

    Raises:
        ValueError: If the results have no world step measurements.
    """
    steps = _measured(results, WORLD_STEP)
    if not steps:
        raise ValueError("no world step results: run tests/benchmarks/test_world_speed.py")
    rows = [WORLD_HEADER]
    for bench in steps:
        cars = bench["extra_info"]["cars"]
        seconds = bench["stats"]["median"]
        rows.append(
            f"| {cars:,} | {_milliseconds(seconds)} | {1 / seconds:,.0f} | {cars / seconds:,.0f} "
            f"| {bench['extra_info']['decision_dt'] / seconds:,.0f}x |"
        )
    readings = _measured(results, SENSE)
    if readings:
        rays = readings[0]["extra_info"]["rays"]
        rows += ["", f"**Distance sensors**, {rays} rays per car, read once per step:", ""]
        rows.append(SENSE_HEADER)
        for bench in readings:
            cars = bench["extra_info"]["cars"]
            seconds = bench["stats"]["median"]
            rows.append(f"| {cars:,} | {_milliseconds(seconds)} | {cars / seconds:,.0f} |")
    rows += _environment_rows(results)
    machine = results["machine_info"]
    cpu = machine["cpu"].get("brand_raw") or machine["processor"]
    rows.append(
        f"\nMedians, measured on {cpu} ({machine['system']} {machine['release']}), "
        f"Python {machine['python_version']}, NumPy {machine.get('numpy', 'unknown')}, "
        f"commit {results['commit_info'].get('id', 'unknown')[:7]}."
    )
    return "\n".join(rows)


def _environment_rows(results: Mapping[str, Any]) -> list[str]:
    """The RL environment table, and how much faster one world is than separate environments."""
    together, apart = _measured(results, ONE_WORLD), _measured(results, SEPARATE)
    if not together and not apart:
        return []
    rows = ["", "**RL environment**, every car's step with its observation and reward:", ""]
    rows.append(ENV_HEADER)
    measured = [("One world (`BatchedRacingEnv`)", bench) for bench in together]
    measured += [("Separate (`SyncVectorEnv` of `RacingEnv`)", bench) for bench in apart]
    for kind, bench in measured:
        cars = bench["extra_info"]["cars"]
        seconds = bench["stats"]["median"]
        rows.append(f"| {kind} | {cars:,} | {_milliseconds(seconds)} | {cars / seconds:,.0f} |")
    same = {bench["extra_info"]["cars"]: bench for bench in together}
    for bench in apart:
        cars = bench["extra_info"]["cars"]
        if cars in same:
            speedup = bench["stats"]["median"] / same[cars]["stats"]["median"]
            rows.append(
                f"\nWith {cars:,} cars, one world is {speedup:,.0f} times faster than "
                f"{cars:,} separate environments."
            )
    return rows


def _measured(results: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    """The results of one parametrized benchmark, fewest cars first."""
    found = [bench for bench in results["benchmarks"] if bench["name"].startswith(name)]
    return sorted(found, key=lambda bench: bench["extra_info"]["cars"])


def _milliseconds(seconds: float) -> str:
    return f"{seconds * 1e3:.2f} ms"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python scripts/benchmark_table.py RESULTS.json")
    try:
        print(speed_table(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))))
    except ValueError as error:
        sys.exit(f"{sys.argv[1]}: {error}")
