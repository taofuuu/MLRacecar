# ADR-0002: Build a custom 2D simulator in Python

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** [ADR-0003](0003-layered-architecture-pure-core.md), [ADR-0005](0005-batched-multi-car-simulation.md)

> **In plain words:** We build our own simple 2D racing game in Python instead of using a game engine like Unity. It's less pretty, but it runs very fast (so the AI learns quickly), it's easy to test, and you'll understand every part of it.

## Context

We need a racing simulation that:

- trains RL agents quickly on one PC (laptop CPU + RTX 4060, 8 GB);
- uses tracks the user designs;
- runs headless in CI on Windows and Linux;
- supports many cars, for parallel training now and multi-car racing later;
- is fully understood and explainable by its author, since the project is a portfolio piece.

## Options considered

1. **Custom 2D top-down simulator in Python/NumPy, behind the Gymnasium API.** Full control,
   very fast vectorized stepping, trivial CI. We build physics, rendering, and tooling ourselves.
2. **Unity + ML-Agents.** Best visuals and PhysX physics. Heavy toolchain, C#/Python split,
   slower simulation, hard to test in CI, and the physics isn't ours.
3. **Godot + godot_rl_agents.** Open source and lighter than Unity, but a smaller ecosystem
   and a process bridge between engine and trainer.
4. **MuJoCo / PyBullet (3D).** Research-grade physics, but modelling cars and tracks takes a
   lot of time that isn't the focus of this project.
5. **Gymnasium `CarRacing-v3`.** Ready-made, but pixel-based, with a fixed track generator,
   no editor, and Box2D physics we don't control.

## Decision

Build a custom top-down 2D simulator in Python (NumPy), exposed through Gymnasium (and
PettingZoo for multi-agent), with pygame for rendering.

## Consequences

- **Positive:** thousands of steps per second, so you can iterate on rewards in minutes, not
  hours. Every line is testable and explainable. Easy to run in CI.
- **Negative / costs:** we own physics, rendering, and editor code, and milestones keep that
  scope under control. Visuals are 2D (a 3D viewer is a possible later add-on).
- **Follow-ups:** M1 (tracks), M2 (simulation), M3 (environment).
