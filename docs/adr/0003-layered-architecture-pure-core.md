# ADR-0003: Layered architecture with a pure NumPy simulation core

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** [architecture.md §2](../architecture.md#2-layers-and-dependency-rules), ticket M0-6

> **In plain words:** The code is organized like floors of a building. The bottom floor (physics and race rules) uses nothing but math: no graphics, no AI libraries. Upper floors may use lower ones, never the other way around, and a tool checks this automatically. That keeps the physics easy to test, and it could even run in a web browser later.

## Context

The same simulation must serve an editor, a desktop viewer, headless training, evaluation,
and later a browser demo. If physics code imports pygame, or the environment imports
PyTorch, every consumer pays for every dependency, tests need a display or GPU, and the
browser demo becomes a rewrite.

## Options considered

1. **Flat package, import anything from anywhere.** Fast to start, but it grows into a tangle.
2. **Layered packages with enforced rules.** A little more structure up front; the rules are
   checked automatically.

## Decision

Organize `mlracecar` into layers (`core` → `config`/`io` → `env`/`render` → `agents` →
`training`/`editor` → `cli`). Dependencies only point downward. `core` may import only the
standard library and NumPy. The rules are enforced by import-linter in CI and pre-commit.
Heavy dependencies (`pygame`, `torch`, `stable-baselines3`) are optional extras.

## Consequences

- **Positive:** `core` is deterministic, fast to test, and needs no display or GPU. It can
  run in a browser via Pyodide. CI installs stay small. Boundaries are visible to reviewers.
- **Negative / costs:** some mapping code between pydantic config models and plain core
  dataclasses; occasional friction when a "quick" import would cross a boundary.

## Amendments

- **2026-10-06 (#6):** import-linter enforces the layer order. The "`core` imports only the
  standard library and NumPy" rule is an allow-list, and import-linter's contracts are
  deny-lists (they can only forbid named packages, so a newly added dependency would slip
  through). That rule is therefore enforced by `tests/unit/test_architecture.py`, which scans
  every import in `core` and also runs in pre-commit and CI.
