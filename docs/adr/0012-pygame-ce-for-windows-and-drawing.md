# ADR-0012: pygame-ce for windows and drawing

- **Status:** Accepted
- **Date:** 2026-10-07
- **Related:** tickets #14 (track editor), M2-6 (race window); ADR-0002 (custom Python
  simulator), ADR-0003 (layered architecture)

> **In plain words:** The editor and the race window need a library that opens a window, reads
> the mouse and keyboard, and draws shapes. We use **pygame-ce**, the actively maintained
> edition of the well-known pygame library. Only the drawing code may use it: everything else
> runs without a screen, and a check fails the build if that ever changes.

## Context

The architecture planned "pygame" for rendering ([§2](../architecture.md#2-layers-and-dependency-rules),
rule 5). The editor (#14) is the first code that needs a window. There are two editions of
pygame today, installed under different names but both imported as `pygame`:

- **pygame**: the original. Its latest release, 2.6.1, is from September 2024.
- **pygame-ce** ("community edition"): a fork by most of pygame's active contributors. It
  releases every few months, ships wheels for new Python versions quickly, has complete type
  hints (we run mypy strict), and adds drawing functions such as anti-aliased circles.

Training must stay headless (rule 3), and the editor's logic must be testable without a
display (#14's acceptance criteria).

## Options considered

1. **pygame**: the best-known name, but releases have slowed and its type hints are
   incomplete.
2. **pygame-ce**: the same API, actively maintained, fully typed. The two can't be installed
   side by side, which only matters to someone who already has pygame installed.
3. **A different toolkit** (pyglet, arcade, Qt): more features, but a different API from the
   one the plan and its tutorials assume, and more than a 2D editor and top-down race need.

## Decision

Use **pygame-ce**, installed through the `render` extra (`pip install 'mlracecar[render]'`) and
the `dev` group, so contributors and CI always have it.

Only drawing code imports it: `mlracecar.render.drawing`, and the editor's view and window. An
import-linter contract forbids pygame everywhere else, including through indirect imports: the
core, config, io, env, agents, training, the camera, and the editor's draft and controller.

## Consequences

- **Positive:** a maintained dependency with types that mypy can check. The editor's
  logic, the camera, and the whole simulation stay testable without a screen, and the build
  enforces it. Tests that do need pygame run off-screen with SDL's dummy video driver, on
  Linux and Windows CI alike.
- **Negative / costs:** contributors with the original pygame installed in the same
  environment must remove it first. `racecar edit` without the `render` extra exits with a
  message that says how to install it, instead of a traceback.
- **Follow-ups:** the race window (M2-6) reuses `mlracecar.render.camera` and
  `mlracecar.render.drawing`.
