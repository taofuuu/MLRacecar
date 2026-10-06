# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `mlracecar.core.geometry`: vectorized 2D geometry (cross product, rotation, angle wrapping,
  segment intersection, raycasts, projection onto a polyline, self-intersection detection),
  property-tested with Hypothesis.
- `mlracecar.core.track.spline`: closed C2 cubic spline through control points (ADR-0010) with
  even arc-length resampling, unit tangents and normals, and signed curvature.
- `mlracecar.core.track.model.Track`: road widths blended smoothly between control points, left
  and right edges, evenly spaced checkpoints with the start/finish line at control point 0,
  poses at any distance along the track, and a staggered two-wide starting grid.
- `mlracecar.core.track.validation.validate`: checks a track and reports every problem with a
  severity, a stable code, a plain-language message, and a location (control point, stretch of
  road, or spot) for the editor to highlight. Rules: too few points, invalid or coinciding
  points, non-positive widths, too narrow, too short, edges folding in tight bends, inside
  edges coming to a sharp point (warning), bends too tight to steer (warning), the track
  crossing itself, and parts of the road overlapping. Each bend, and each run of too-narrow
  points, is reported once.
- A realistic 3.5 km GP-style test circuit (`tests/circuits.py`) that validation is checked
  against, both as designed and pushed past its limits.
- `mlracecar.core.geometry.polyline_crossings`, and a bounding-box broad phase for all crossing
  searches: about 46x faster on a 1.5 km track (validation takes ~33 ms there).
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
