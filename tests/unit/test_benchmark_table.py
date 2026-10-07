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
        "Median of 200 steps on Some CPU (Linux 6.8), Python 3.12.3, NumPy 2.5.3, commit 0c09c51."
    )


def test_other_benchmarks_are_left_out() -> None:
    render = {"name": "test_draw_sixteen_cars[follow]", "stats": {"median": 0.007}}
    table = speed_table(results(render, world_step(1, 0.0005)))

    assert "draw" not in table
    assert len(table.splitlines()) == 5  # header, divider, one row, blank, where


def test_results_without_world_steps_are_refused() -> None:
    with pytest.raises(ValueError, match="no world step results"):
        speed_table(results())
