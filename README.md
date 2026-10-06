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

> **Status: planning.** Design and [backlog](https://github.com/users/taofuuu/projects/3) are done; implementation starts with
> [milestone M0](docs/roadmap.md#m0-foundations).

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
uv run racecar edit tracks/gp-circuit.json
uv run racecar check tracks/gp-circuit.json
```

`racecar edit` opens the track editor: click to add points, the road appears as you draw, and
any problems are marked in red or orange. Ctrl+S saves. Press **H** in the editor for all the
controls. `racecar check` reads a track file and runs the
track checks on it. The sample tracks are in [`tracks/`](tracks/); the format is described in
[docs/track-format.md](docs/track-format.md).

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
