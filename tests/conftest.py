"""Shared pytest configuration for the whole test suite."""

from pathlib import Path

import pytest

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark everything under tests/benchmarks so it is skipped unless `-m benchmark` is given."""
    for item in items:
        if BENCHMARKS_DIR in item.path.parents:
            item.add_marker(pytest.mark.benchmark)
