"""Shared pytest configuration for the whole test suite."""

import os
from pathlib import Path

import pytest
from hypothesis import settings

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"

# pygame windows open off-screen, so the editor and renderer tests run anywhere, CI included.
# Set SDL_VIDEODRIVER yourself (e.g. to "windows") to watch them.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# No per-example time limit: CI machines (especially Windows) are slower and would make
# property tests flaky. print_blob shows how to replay any failure exactly.
settings.register_profile("project", deadline=None, print_blob=True)
settings.load_profile("project")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark everything under tests/benchmarks so it is skipped unless `-m benchmark` is given."""
    for item in items:
        if BENCHMARKS_DIR in item.path.parents:
            item.add_marker(pytest.mark.benchmark)
