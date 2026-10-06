# Vision

> **In plain words:** We're building a 2D racing game where you draw the track and an AI
> learns to drive it. First we make the code clean and solid (to show software-engineering
> skill), then we go deep on the AI side (to show machine-learning skill). This page lists
> what "success" means, in numbers. Unfamiliar terms are in the [glossary](glossary.md).

## Pitch

**Draw a race track. Watch an AI learn to master it. Then try to beat it.**

MLRacecar is a top-down 2D racing simulator written from scratch in Python. You design
tracks in a visual editor, a reinforcement-learning (RL) agent learns to drive them by
trial and error, and you can race against it on your desktop or in the browser.

## Why this project exists

This is a portfolio project built in two deliberate phases:

1. **Software engineering first.** A clean, layered, tested, reproducible system: the kind
   of codebase a team could pick up and extend.
2. **ML/RL depth second.** Once the foundation is solid, go deep on algorithms (our own
   PPO implementation), experiments, and analysis.

The order matters: an ML experiment is only as trustworthy as the software running it.
A deterministic simulator, versioned configs, and reproducible runs come before tuning.

## Goals

| #  | Goal                                   | Measured by                                                    | Milestone |
|----|----------------------------------------|----------------------------------------------------------------|-----------|
| G1 | Design any track visually              | A new valid track can be built in under 5 minutes             | M1        |
| G2 | Drive it yourself                      | Keyboard driving with lap timing                               | M2        |
| G3 | An agent learns to lap your track      | ≥ 95% lap completion over 20 evaluation episodes               | M4        |
| G4 | The agent generalizes to unseen tracks | ≥ 80% lap completion on 20 held-out generated tracks           | M5        |
| G5 | Race the AI                            | Human-vs-AI race mode with results                             | M5        |
| G6 | Anyone can try it in 10 seconds        | Public browser demo link                                       | M6        |
| G7 | Cars race each other                   | Self-play agent wins > 70% of head-to-heads vs. the M5 agent   | M7        |
| G8 | Demonstrate ML depth                   | Own PPO matches SB3 within confidence intervals; ablation study | M8        |

## Engineering quality bar

These apply to every milestone, not just the last one:

- CI is green on **Windows and Linux** for every merged PR; `main` is always releasable.
- Test coverage ≥ 90% for `core`, ≥ 80% overall.
- Same seed + same config → identical simulation results (bitwise, on the same platform).
- Every training result is traceable to a run directory (resolved config, git SHA, seed).
- Every significant decision is recorded as an [ADR](adr/).
- Fresh clone → driving your own track in under 5 minutes, one command per step.
- Performance is measured, not guessed: throughput benchmarks are published in the README.

## Non-goals

- 3D graphics or photorealism. A 3D *viewer* could come later; the simulation stays 2D.
- Real-world deployment (sim-to-real on a physical car).
- Online multiplayer or networking.
- Commercial-sim physics fidelity. Physics must be plausible, documented, and tested,
  not certified.
- Supporting every RL library. One library baseline (Stable-Baselines3) and one in-house
  implementation.

## Audiences

| Who                     | Time they give it | What they must see                                              |
|-------------------------|-------------------|-----------------------------------------------------------------|
| Recruiter               | 30 seconds        | README hero GIF, one-line pitch, live demo link                 |
| Hiring engineer         | 5 minutes         | Architecture doc, ADRs, CI badges, clean PR history             |
| Interviewer (deep dive) | 30+ minutes       | Code quality, tests, experiment log, results with error bars    |
| You                     | Ongoing           | A codebase that is pleasant to extend and explainable line by line |

## Principles

1. **Pure, deterministic core.** Physics and race rules are plain NumPy, with no I/O,
   rendering, or global state.
2. **Vertical slices.** Every milestone ends with something you can demo.
3. **Reproducible by default.** Configs are typed and saved with every run; randomness is
   seeded explicitly.
4. **Measure before optimizing.** Benchmarks first, then optimize the hot spots.
5. **Docs as code.** Architecture, decisions, and experiments live in the repo and are
   reviewed like code.
6. **No black boxes.** If we can't explain it in an interview, we don't own it yet. Library
   pieces get replaced with our own where that adds real understanding (e.g. PPO in M8).
