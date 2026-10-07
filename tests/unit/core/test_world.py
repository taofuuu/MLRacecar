"""Tests for mlracecar.core.world."""

import pytest

from mlracecar.core.world import Timing


def test_timing_turns_rates_into_step_lengths() -> None:
    timing = Timing(physics_hz=120, action_repeat=6)

    assert timing.dt == pytest.approx(1 / 120)
    assert timing.decision_dt == pytest.approx(0.05)  # 20 decisions a second
