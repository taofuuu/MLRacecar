# ADR-0007: Python toolchain: uv, ruff, mypy, pytest, GitHub Actions

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** milestone M0

> **In plain words:** The developer tools we use: uv installs Python and libraries, ruff checks code style, mypy catches type mistakes, pytest runs the tests, and GitHub Actions runs all of them automatically on every change.

## Context

We want a modern, fast, reproducible toolchain that behaves the same on Windows (where we
develop) and Linux (CI), and that a reviewer recognizes as current best practice.

## Decision

| Concern               | Tool                                   | Why                                                              |
|-----------------------|----------------------------------------|------------------------------------------------------------------|
| Environments and deps | **uv** (`pyproject.toml` + `uv.lock`)  | Very fast, lockfile-based, manages the Python version too        |
| Python version        | **3.12**                               | Installed locally; supported by Pyodide, PyTorch, and SB3        |
| Lint and format       | **ruff**                               | One fast tool replacing flake8, isort, and black                 |
| Type checking         | **mypy** (strict on `core`)            | The most widely recognized checker; strictness where it pays off |
| Tests                 | **pytest**, Hypothesis, pytest-cov, pytest-benchmark | The standard; property tests suit geometry and physics |
| Architecture rules    | **import-linter**                      | Enforces [ADR-0003](0003-layered-architecture-pure-core.md)      |
| Git hooks             | **pre-commit**                         | Same checks locally and in CI                                    |
| CI/CD                 | **GitHub Actions** (Windows + Ubuntu)  | Lives next to the code, issues, and project board                |
| CLI                   | **Typer** + Rich                       | Type-hint-driven commands, readable output                       |
| Docs site             | **MkDocs Material**                    | Markdown docs we already write, published to GitHub Pages        |

## Consequences

- **Positive:** a single `uv sync` sets up everything; the lockfile makes builds reproducible;
  checks are fast enough to run on every commit.
- **Negative / costs:** uv is newer than pip or Poetry (mature enough in 2026). PyTorch with
  CUDA needs an explicit package index in `pyproject.toml` (ticket M4-1).
