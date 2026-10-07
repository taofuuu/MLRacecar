# Tests

> **In plain words:** Each folder holds a different kind of test. `uv run pytest` runs the fast
> ones; slow, GPU, and speed-measurement tests only run when you ask for them.

| Folder         | What goes here                                                        | Runs by default |
|----------------|-----------------------------------------------------------------------|-----------------|
| `unit/`        | One small piece in isolation (geometry, dynamics, rewards, CLI)       | Yes             |
| `integration/` | Several pieces together (a full episode in the environment)           | Yes             |
| `regression/`  | Saved "golden" results that future runs must match                    | Yes             |
| `benchmarks/`  | Speed measurements (pytest-benchmark)                                  | No: `-m benchmark` |

## Golden trajectories

`regression/golden/*.npz` are recorded runs of the simulation: scripted cars driving the
sample tracks through the car physics and the race rules, recorded after every driver
decision (the scenarios are in `golden.py`). `regression/test_golden.py` runs them again and
compares. Tiny differences (about a millionth) are allowed, because Windows and Linux compute
sin, cos, and friends very slightly differently; anything bigger fails with a summary of
what changed, when, and by how much.

If you changed the physics, a car setting, or the race rules **on purpose**, record them again
and say what changed and why in the pull request:

```bash
uv run python scripts/update_golden.py
```

Markers: `slow` (more than a few seconds) and `gpu` (needs CUDA) are skipped unless selected,
e.g. `uv run pytest -m slow`. See [architecture.md §6](../docs/architecture.md#6-testing-strategy)
for the full testing strategy.
