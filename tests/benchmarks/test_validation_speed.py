"""How long validating a track takes. The editor re-validates on every edit, so it must stay fast.

Run with `uv run pytest -m benchmark --no-cov`.
"""

import numpy as np
from pytest_benchmark.fixture import BenchmarkFixture

from mlracecar.core.track.validation import validate


def test_validate_a_one_and_a_half_kilometre_track(benchmark: BenchmarkFixture) -> None:
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    points = np.column_stack([300 * np.cos(angles), 150 * np.sin(angles)])  # about 1.5 km
    issues = benchmark(validate, points, np.full(len(points), 12.0))
    assert issues == []
