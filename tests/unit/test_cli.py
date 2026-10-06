"""Smoke tests for the `racecar` command line."""

import re

from typer.testing import CliRunner

import mlracecar
from mlracecar.cli import app

runner = CliRunner()

# Typer colors its help output when it detects CI (e.g. GITHUB_ACTIONS is set). The color
# codes split words like "--version", so tests compare against the plain text.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def test_version_flag_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"mlracecar {mlracecar.__version__}"


def test_version_follows_pep_440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", mlracecar.__version__)


def test_no_arguments_shows_help() -> None:
    output = plain(runner.invoke(app, []).output)

    assert "Usage" in output
    assert "--version" in output
