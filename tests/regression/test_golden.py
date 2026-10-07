"""Regression tests: the simulation must keep matching its golden trajectories (see golden.py)."""

from dataclasses import replace

import numpy as np
import pytest

import mlracecar.core.vehicle.dynamics as dynamics
from golden import SCENARIOS, Recording, Scenario, differences, read, record
from mlracecar.config.models import RacecarConfig

BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_the_simulation_still_matches_its_golden_run(scenario: Scenario) -> None:
    summary = differences(scenario, read(scenario), record(scenario))

    assert summary is None, summary


def test_the_same_run_twice_is_identical_to_the_last_digit() -> None:
    scenario = BY_NAME["mixed-batch"]  # four cars, off the road, the wrong way: the busiest

    first, second = record(scenario), record(scenario)

    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def test_a_changed_physics_constant_fails_with_a_readable_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = BY_NAME["straight-line"]
    monkeypatch.setattr(dynamics, "AIR_DENSITY", 1.3)  # thicker air: more drag

    summary = differences(scenario, read(scenario), record(scenario))

    assert summary is not None
    lines = summary.splitlines()
    assert lines[0] == "The 'straight-line' run no longer matches its golden trajectory:"
    assert any(line.startswith("  vx: first differs at step ") for line in lines)
    assert "    uv run python scripts/update_golden.py" in lines
    assert lines[-1] == "and say why in the pull request."


def test_a_changed_car_setting_is_caught_too() -> None:
    scenario = BY_NAME["straight-line"]
    stronger = RacecarConfig.model_validate({"vehicle": {"max_power": 260}})

    summary = differences(scenario, read(scenario), record(replace(scenario, config=stronger)))

    assert summary is not None
    assert "  vx: first differs" in summary


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #


def recording(**columns: object) -> Recording:
    base: Recording = {"time": np.array([0.05, 0.1]), "x": np.array([[1.0], [2.0]])}
    base.update({name: np.asarray(value) for name, value in columns.items()})
    return base


def compare(golden: Recording, now: Recording) -> str | None:
    return differences(SCENARIOS[0], golden, now)


def test_noise_far_below_what_physics_changes_cause_is_ignored() -> None:
    noisy = recording(x=np.array([[1.0 + 1e-9], [2.0 - 1e-9]]))

    assert compare(recording(), noisy) is None


def test_angles_a_turn_apart_count_as_the_same() -> None:
    golden = recording(yaw=np.array([[np.pi - 1e-12], [0.0]]))
    now = recording(yaw=np.array([[-np.pi + 1e-12], [0.0]]))

    assert compare(golden, now) is None


@pytest.mark.parametrize(
    ("now", "says"),
    [
        (recording(x=np.array([[1.0], [2.5]])),
         "x: first differs at step 1 (t = 0.10 s), car 0: golden 2, now 2.5 (0.5 apart); "
         "at most 0.5 apart"),
        (recording(laps=np.array([[0], [1]])), "laps: only in the new run"),
        (recording(x=np.array([[1.0, 1.0], [2.0, 2.0]])), "x: shape (2, 2), golden (2, 1)"),
    ],
)  # fmt: skip
def test_each_kind_of_difference_is_named(now: Recording, says: str) -> None:
    summary = compare(recording(), now)

    assert summary is not None
    assert f"  {says}" in summary.splitlines()[1:]


def test_whole_number_quantities_must_match_exactly() -> None:
    golden = recording(checkpoint=np.array([[3], [4]]))

    summary = compare(golden, recording(checkpoint=np.array([[3], [5]])))

    assert summary is not None
    assert "checkpoint: first differs at step 1 (t = 0.10 s), car 0: golden 4, now 5 (" in summary


def test_one_dimensional_quantities_name_the_entry() -> None:
    golden = recording(lap_time=np.array([51.1, 50.9]))

    summary = compare(golden, recording(lap_time=np.array([51.1, 52.0])))

    assert summary is not None
    assert "lap_time: first differs at entry 1: golden 50.9, now 52" in summary
