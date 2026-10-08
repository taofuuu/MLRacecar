"""The README's speed table, made from pytest-benchmark results by scripts/benchmark_table.py."""

import runpy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_table.py"
speed_table = runpy.run_path(str(SCRIPT))["speed_table"]


def world_step(cars: int, median: float) -> dict[str, Any]:
    return {
        "name": f"test_world_step[{cars}]",
        "extra_info": {"cars": cars, "decision_dt": 0.05},
        "stats": {"median": median, "rounds": 200},
    }


def sense(cars: int, median: float) -> dict[str, Any]:
    return {
        "name": f"test_sense[{cars}]",
        "extra_info": {"cars": cars, "rays": 15},
        "stats": {"median": median},
    }


def env_step(name: str, cars: int, median: float) -> dict[str, Any]:
    return {"name": f"{name}[{cars}]", "extra_info": {"cars": cars}, "stats": {"median": median}}


def results(*benchmarks: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmarks": list(benchmarks),
        "machine_info": {
            "cpu": {"brand_raw": "Some CPU"},
            "processor": "x86",
            "system": "Linux",
            "release": "6.8",
            "python_version": "3.12.3",
            "numpy": "2.5.3",
        },
        "commit_info": {"id": "0c09c514e75dc11c0f752bf3b94ae8c1dadfacc1"},
    }


def test_each_car_count_gets_a_row_in_order_with_its_speeds() -> None:
    table = speed_table(results(world_step(1024, 0.004), world_step(1, 0.0005)))

    assert table.splitlines()[2:4] == [
        "| 1 | 0.50 ms | 2,000 | 2,000 | 100x |",
        "| 1,024 | 4.00 ms | 250 | 256,000 | 12x |",
    ]


def test_it_says_where_the_numbers_were_measured() -> None:
    table = speed_table(results(world_step(1, 0.0005)))

    assert table.splitlines()[-1] == (
        "Medians, measured on Some CPU (Linux 6.8), Python 3.12.3, NumPy 2.5.3, commit 0c09c51."
    )


def test_the_sensors_get_their_own_table_when_they_were_measured() -> None:
    table = speed_table(results(world_step(1, 0.0005), sense(64, 0.0016), sense(1, 0.0001)))

    assert table.splitlines()[3:10] == [
        "",
        "**Distance sensors**, 15 rays per car, read once per step:",
        "",
        "| Cars | Time to read every ray | Car readings per second |",
        "|-----:|-----------------------:|------------------------:|",
        "| 1 | 0.10 ms | 10,000 |",
        "| 64 | 1.60 ms | 40,000 |",
    ]


def test_the_environment_gets_its_own_table_and_the_speed_up() -> None:
    table = speed_table(
        results(
            world_step(1, 0.0005),
            env_step("test_separate_environments", 64, 0.064),
            env_step("test_one_world", 64, 0.004),
            env_step("test_one_world", 1, 0.001),
        )
    )

    lines = table.splitlines()
    start = lines.index("| Environment | Cars | Time per step | Car-steps per second |")
    assert lines[start + 2 : start + 5] == [
        "| One world (`BatchedRacingEnv`) | 1 | 1.00 ms | 1,000 |",
        "| One world (`BatchedRacingEnv`) | 64 | 4.00 ms | 16,000 |",
        "| Separate (`SyncVectorEnv` of `RacingEnv`) | 64 | 64.00 ms | 1,000 |",
    ]
    assert "With 64 cars, one world is 16 times faster than 64 separate environments." in lines


def test_other_benchmarks_are_left_out() -> None:
    render = {"name": "test_draw_sixteen_cars[follow]", "stats": {"median": 0.007}}
    table = speed_table(results(render, world_step(1, 0.0005)))

    assert "draw" not in table
    assert len(table.splitlines()) == 5  # header, divider, one row, blank, where


def test_results_without_world_steps_are_refused() -> None:
    with pytest.raises(ValueError, match="no world step results"):
        speed_table(results())
