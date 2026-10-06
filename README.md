# MLRacecar

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

## Documentation

| Document                                 | What's inside                                     |
|------------------------------------------|---------------------------------------------------|
| [Vision](docs/vision.md)                 | Goals, non-goals, success metrics, principles     |
| [Architecture](docs/architecture.md)     | System design, layers, data flow, testing strategy |
| [Decision records](docs/adr/)            | Why we chose what we chose                        |
| [Roadmap](docs/roadmap.md)               | Milestones and backlog                            |
| [Contributing](CONTRIBUTING.md)          | Workflow, conventions, Definition of Done         |
| [Glossary](docs/glossary.md)             | Every technical term, explained in plain words    |

## Tech stack

Python 3.12 · NumPy · Gymnasium · PettingZoo · pygame · PyTorch · Stable-Baselines3 ·
pydantic · Typer · uv · ruff · mypy · pytest + Hypothesis · GitHub Actions

## License

[MIT](LICENSE)
