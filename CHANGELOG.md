# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project vision, architecture, ADRs 0001–0009, roadmap, and contribution workflow.
- Backlog definition and an idempotent GitHub seeding script.
- Plain-language summaries on every planning document, plus a glossary.
- Python package scaffold (`src/mlracecar`, uv, Python 3.12) with a `racecar` CLI
  (`racecar --version`).
- Code-quality gates: ruff lint and format, strict mypy, and pre-commit hooks.
- Test framework: pytest with coverage (80% minimum), Hypothesis, pytest-benchmark, and opt-in
  `slow` / `gpu` / `benchmark` markers.
- Continuous integration on GitHub Actions: all pre-commit hooks, plus tests on Ubuntu and
  Windows with a coverage summary. `main` is protected: merging requires a PR with green CI.
- Architecture layer packages (`core`, `config`, `io`, `env`, `render`, `agents`, `training`,
  `editor`) with automated boundary checks: import-linter for the layer order and an
  allow-list test keeping `core` to the standard library and NumPy.
- Issue forms (feature, bug, research spike) that label new tickets and add them to the board,
  and a pull request template with the Definition of Done.
- Documentation website (MkDocs Material) with Mermaid diagrams and an API reference generated
  from docstrings; built in strict mode on every PR and published to GitHub Pages from `main`.
