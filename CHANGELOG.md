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
- Track files (`mlracecar.io.track_file`): versioned JSON (format version 1) with field-level
  error messages, automatic upgrades from older versions, refusal of newer ones, atomic saving,
  one control point per line, and a published JSON Schema (`docs/schemas/`).
- Sample tracks in `tracks/`: an oval and a 3.5 km GP circuit, both free of track-check issues.
- `racecar check <file>`: reads a track file and runs the track checks on it.
- `mlracecar.editor.draft.TrackDraft`: the track editor's headless model. Immutable drafts
  (ADR-0011) with append, insert-into-stretch, move, delete, width, reverse-direction, and
  set-start edits, cached track and validation per draft, and conversion to and from track files.
- The track editor window, `racecar edit [file]`: click to draw a track and watch the road
  appear, shift+click to insert, drag to move, right-click to delete, shift+wheel or `[`/`]`
  for road width, pan and zoom, grid snap (G), fit (F), reverse (R), start line (S), and a help
  panel (H). Opens existing track files; saving comes next.
- `mlracecar.render`: an immutable `Camera` (metres to pixels, pan, zoom around the cursor,
  fit, grid spacing) and pygame drawing of the grid and the track (road, edges, start line,
  direction arrow) that fills only what's on screen, for a few milliseconds per frame at any
  zoom.
- The `render` extra now installs pygame-ce (ADR-0012). An architecture rule keeps pygame out
  of everything that must run headless.
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
