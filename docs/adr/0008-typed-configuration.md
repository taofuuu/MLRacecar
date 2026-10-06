# ADR-0008: Typed configuration with pydantic and YAML

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** ticket M2-1, [ADR-0003](0003-layered-architecture-pure-core.md)

> **In plain words:** Settings (car speed, rewards, training options) live in readable text files. They are checked when loaded, so a typo gives a clear error *before* a long training run starts. The exact settings used are saved with every training run.

## Context

Vehicle parameters, environment settings (sensors, rewards, termination), and training
hyperparameters all need to be configurable, validated early with clear errors, and saved
with every run for reproducibility.

## Options considered

1. **Hydra / OmegaConf.** Powerful composition and multirun sweeps, but a lot of magic,
   weak static typing, and it takes over the application entry point.
2. **Dataclasses + argparse.** Simple and explicit, but no file loading, validation, or
   error reporting without writing it all ourselves.
3. **Pydantic v2 models + YAML files.** Typed validation with path-qualified errors and JSON
   Schema export. Composition is limited to "base file + overrides".

## Decision

Pydantic v2 models live in `mlracecar.config`, YAML files in `configs/`, and CLI overrides
use `--set section.key=value`. Precedence: defaults < file < CLI. The fully resolved config
is written to every run directory. `core` never sees pydantic: config models convert to
frozen dataclasses (`VehicleParams`, …) before reaching the simulation.

## Consequences

- **Positive:** invalid configs fail before training starts, with errors like
  `vehicle.max_steer: must be > 0`. The schema documents itself.
- **Negative / costs:** no built-in hyperparameter sweeps; we'll use Optuna for that (M8-3).
