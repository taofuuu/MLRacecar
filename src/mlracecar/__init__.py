"""MLRacecar: design a race track, train an AI to master it, then race it."""

from importlib.metadata import version

# The version lives in pyproject.toml only; this reads it from the installed package metadata.
__version__ = version("mlracecar")

__all__ = ["__version__"]
