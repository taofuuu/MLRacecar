"""Smoke tests for the `racecar` command line."""

import re

from typer.testing import CliRunner

import mlracecar
from mlracecar.cli import app

runner = CliRunner()


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"mlracecar {mlracecar.__version__}"


def test_version_follows_pep_440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", mlracecar.__version__)


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output
    assert "--version" in result.output
