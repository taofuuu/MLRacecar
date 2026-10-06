# ADR-0005: Batched struct-of-arrays simulation, multi-car from day one

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** [architecture.md §4.8](../architecture.md#48-environments), tickets M2-4, M3-5, M7-1

> **In plain words:** Instead of updating cars one at a time, we keep all cars' data in big lists and update them all in one go. That's much faster in Python, and it means "many cars on the track" is built in from day one: useful for training many cars in parallel now, and for real races later.

## Context

RL training needs many environment steps; PPO typically uses 16–128 parallel environments.
The roadmap also includes multi-car racing (M7). A `Car` object with per-car Python methods
is easy to read but slow: Python overhead per car per tick dominates, and retrofitting
multi-car support later means a rewrite.

## Options considered

1. **Object per car, Python loops.** Most readable, slowest, and multi-car needs a redesign.
2. **One environment per process (`SubprocVecEnv`).** Standard, but adds IPC overhead and
   memory, and doesn't help multi-car racing.
3. **One world simulating N cars as struct-of-arrays.** Every physics update is a single
   NumPy operation over all cars. Single-agent, batched, and multi-agent environments
   become configurations of the same world.

## Decision

`World` simulates N cars using struct-of-arrays state (`x[N]`, `y[N]`, `yaw[N]`, …).

- Single-agent env: N = 1.
- Batched env: N independent "ghost" cars that don't collide = N parallel environments.
- Multi-agent env (M7): N cars with collisions enabled.

## Consequences

- **Positive:** large speedups from vectorization (to be measured in M2-9 and M3-5).
  Multi-car racing becomes a feature (collisions + observations) rather than a rewrite.
- **Negative / costs:** vectorized code is harder to read than per-car objects. Mitigated by
  clear docs, small functions, and equivalence tests (one car simulated alone ≡ the same car
  inside a batch of 1024).
