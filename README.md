# MLRacecar

[![CI](https://github.com/taofuuu/MLRacecar/actions/workflows/ci.yml/badge.svg)](https://github.com/taofuuu/MLRacecar/actions/workflows/ci.yml)
[![Docs](https://github.com/taofuuu/MLRacecar/actions/workflows/docs.yml/badge.svg)](https://taofuuu.github.io/MLRacecar/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Draw a race track. Watch an AI learn to master it. Then try to beat it.**

MLRacecar is a top-down 2D racing simulator built from scratch in Python. It comes with a
visual track editor and a reinforcement-learning pipeline that teaches cars to drive any
track you design.

> **Status: in development.** Milestones M0 (foundations) and M1 (tracks) are done, and so is
> M2 (drivable simulation) apart from the optional tire-slip car model: you can draw a track in
> the editor and drive it with the keyboard. Next up is M3, the reinforcement-learning
> environment. See the [roadmap](docs/roadmap.md) and the
> [project board](https://github.com/users/taofuuu/projects/3).

## Planned features

- **Track editor:** design circuits with splines; live validation; versioned JSON files
- **Vectorized physics:** kinematic and dynamic bicycle models simulating many cars at once
- **RL environments:** Gymnasium-compatible, with lidar-style raycast sensors and configurable rewards
- **Reproducible training:** typed configs, self-describing run directories, experiment tracking
- **Play against the AI:** race trained agents or chase their ghost laps
- **Browser demo:** watch agents race from a link, no install needed
- **Multi-car racing:** collisions, overtaking, and self-play training
- **From-scratch PPO:** benchmarked against Stable-Baselines3 with proper statistics

## Try it

```bash
uv sync
uv run racecar drive tracks/gp-circuit.json
uv run racecar edit tracks/gp-circuit.json
uv run racecar check tracks/gp-circuit.json
uv run racecar config
```

`racecar drive` puts you behind the wheel: arrow keys or WASD to drive, **R** to restart,
**C** to change the camera, **4** to see the distance sensors the AI will drive by
([docs/sensors.md](docs/sensors.md)). The panel shows your speed, lap, and lap times; a lap only counts
if you stay on the road and pass every checkpoint in order. `racecar edit` opens the track editor: click to add points, the road appears as you draw, and
any problems are marked in red or orange. Ctrl+Z undoes, Ctrl+S saves. Press **H** in the editor for all the
controls. `racecar check` reads a track file and runs the
track checks on it. The sample tracks are in [`tracks/`](tracks/); the format is described in
[docs/track-format.md](docs/track-format.md). `racecar config` shows the settings the
simulation will run on, such as the car's weight and power; see
[docs/configuration.md](docs/configuration.md) for how to change them.

## Speed

The simulation moves every car at once with NumPy arrays, so a thousand cars cost only a few
times more than one. One step is one driver decision: 1/20 s of racing, with six physics
updates and the race rules for every car. A car-step is one car driving one step: one
experience for the AI to learn from. The AI also reads each car's distance sensors once per
step.

| Cars | Time per step | Steps per second | Car-steps per second | Faster than real time |
|-----:|--------------:|-----------------:|---------------------:|----------------------:|
| 1 | 0.40 ms | 2,501 | 2,501 | 125x |
| 64 | 0.50 ms | 1,985 | 127,009 | 99x |
| 1,024 | 2.35 ms | 426 | 435,847 | 21x |

**Distance sensors**, 15 rays per car, read once per step:

| Cars | Time to read every ray | Car readings per second |
|-----:|-----------------------:|------------------------:|
| 1 | 0.08 ms | 12,937 |
| 64 | 1.35 ms | 47,474 |
| 1,024 | 25.67 ms | 39,892 |

Medians, measured on 12th Gen Intel(R) Core(TM) i5-12400F (Windows 11), Python 3.12.3, NumPy 2.5.3, commit 0776a40.

Measure it yourself. Runs back to back agree within about 10%, but on different days the same
computer has measured up to 30% apart, depending on what else it is doing:

```bash
uv run pytest -m benchmark --no-cov --benchmark-json=benchmark.json
uv run python scripts/benchmark_table.py benchmark.json
```

Every push to `main` measures again on GitHub's machines: the
[Benchmarks workflow](https://github.com/taofuuu/MLRacecar/actions/workflows/benchmarks.yml)
shows the tables in each run's summary and keeps the full results as a download.

## Documentation

| Document                                 | What's inside                                     |
|------------------------------------------|---------------------------------------------------|
| [Vision](docs/vision.md)                 | Goals, non-goals, success metrics, principles     |
| [Architecture](docs/architecture.md)     | System design, layers, data flow, testing strategy |
| [Decision records](docs/adr/)            | Why we chose what we chose                        |
| [Roadmap](docs/roadmap.md)               | Milestones and backlog                            |
| [Contributing](CONTRIBUTING.md)          | Workflow, conventions, Definition of Done         |
| [Glossary](docs/glossary.md)             | Every technical term, explained in plain words    |

All of it is also published as a website, with search and diagrams:
**[taofuuu.github.io/MLRacecar](https://taofuuu.github.io/MLRacecar/)**

## Tech stack

Python 3.12 · NumPy · Gymnasium · PettingZoo · pygame · PyTorch · Stable-Baselines3 ·
pydantic · Typer · uv · ruff · mypy · pytest + Hypothesis · GitHub Actions

## License

[MIT](LICENSE)
