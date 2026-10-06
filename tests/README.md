# Tests

> **In plain words:** Each folder holds a different kind of test. `uv run pytest` runs the fast
> ones; slow, GPU, and speed-measurement tests only run when you ask for them.

| Folder         | What goes here                                                        | Runs by default |
|----------------|-----------------------------------------------------------------------|-----------------|
| `unit/`        | One small piece in isolation (geometry, dynamics, rewards, CLI)       | Yes             |
| `integration/` | Several pieces together (a full episode in the environment)           | Yes             |
| `regression/`  | Saved "golden" results that future runs must match                    | Yes             |
| `benchmarks/`  | Speed measurements (pytest-benchmark)                                  | No: `-m benchmark` |

Markers: `slow` (more than a few seconds) and `gpu` (needs CUDA) are skipped unless selected,
e.g. `uv run pytest -m slow`. See [architecture.md §6](../docs/architecture.md#6-testing-strategy)
for the full testing strategy.
