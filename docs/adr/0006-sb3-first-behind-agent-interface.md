# ADR-0006: Stable-Baselines3 first, behind an `Agent` interface

- **Status:** Accepted
- **Date:** 2026-10-06
- **Related:** tickets M4-2, M4-8, M8-1, M8-2

> **In plain words:** For the first AI driver we use a ready-made, well-tested learning algorithm (Stable-Baselines3) instead of writing our own. If the car fails to learn, we know the problem is in our game or rewards, not the algorithm. Later (M8) we write our own and compare the two.

## Context

The author is new to RL. When an agent fails to learn, the cause can be the environment,
the reward, the observations, the hyperparameters, or the algorithm itself. With a
home-made algorithm *and* a home-made environment, these failure sources can't be told apart.

## Options considered

1. **Own PPO first.** Maximum learning, but debugging is confounded and the MVP is delayed.
2. **Stable-Baselines3 (SB3).** Mature, well-tested PyTorch implementations (PPO, SAC) with
   a Gymnasium API.
3. **CleanRL.** Excellent single-file reference implementations; best for reading, not for
   importing as a library.
4. **RLlib.** Distributed and powerful; heavy and overkill for one machine.

## Decision

Use SB3 PPO for the first learning agent, wrapped in our own `Agent` protocol so nothing
outside `agents/` and `training/` depends on SB3. In M8 we implement PPO ourselves (using
CleanRL and the original paper as references) and benchmark it against SB3 on the same
environment across multiple seeds.

## Consequences

- **Positive:** if SB3 can't learn our task, the problem is in our environment or reward,
  not the algorithm. Fast path to the MVP. SB3 later serves as a trusted baseline for our
  own PPO ("matches SB3 within confidence intervals" is a strong, verifiable claim).
- **Negative / costs:** PyTorch is a large dependency, so it's isolated in the optional
  `train` extra. Model saving needs our own model-card metadata on top of SB3's format.
