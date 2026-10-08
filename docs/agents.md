# Trained agents

> **In plain words:** A trained AI driver is saved as a folder with two files: the network
> itself, and a "model card" describing exactly what it was trained to see, with which settings,
> with which code and library versions. When the AI is loaded to drive, the card is checked
> against the game it's about to play. If the game would show it different numbers (say a
> sensor setting changed), loading stops with a message saying exactly what's different,
> instead of the AI quietly driving badly.

Trained models come from [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3),
and are used through MLRacecar's own `Agent` protocol ([ADR-0006](adr/0006-sb3-first-behind-agent-interface.md)),
so nothing else depends on SB3. Everything here needs the training libraries:
`uv sync --extra train` (or `--extra train-cpu`).

## `SB3Agent`

`mlracecar.agents.sb3.SB3Agent` drives with a trained model: `act(observations)` takes a batch
of observations, shape `(cars, 31)`, and returns `[steer, pedal]` for each car, shape
`(cars, 2)`.

- **Deterministic** (the default): it acts on the policy's best guess, its mean action, so the
  same observation always gives the same action. Use this to evaluate and race.
- **Sampling** (`deterministic=False`): it draws actions the way training does. `reset(seed)`
  makes the draws repeat; it seeds PyTorch's, NumPy's, and Python's global random numbers,
  which SB3 samples from.

```python
from mlracecar.agents.sb3 import SB3Agent, make_model_card

agent = SB3Agent(model, make_model_card(model, config, env.observations.spec))
agent.save("models/first-lap")  # model.zip + model_card.json

agent = SB3Agent.load("models/first-lap", env.observations.spec)  # checks the card first
actions = agent.act(observations)
```

`load` runs the network on the CPU by default, which is faster than the GPU for networks this
small; pass `device="cuda"` to use the GPU.

## The model card

`model_card.json` (`mlracecar.io.model_card`) is versioned JSON, readable in any editor:

| Field | What |
|-------|------|
| `algorithm` | `PPO` or `SAC`: what to load `model.zip` with. |
| `created` | When it was saved (UTC). |
| `observation`, `observation_digest` | The full description of the [observation](observations.md) it learned from, and its digest. |
| `action` | `steer` and `pedal`, each from −1 to 1. |
| `config` | Every setting the environment ran with, so it can be rebuilt exactly. |
| `versions` | Python, mlracecar, NumPy, Gymnasium, PyTorch, and Stable-Baselines3. |
| `git` | The commit the code was at, and whether it had uncommitted changes (`null` outside a git checkout). |
| `metadata` | Anything else worth keeping, such as how many steps it trained for. |

Reading a card checks every field and lists every problem at once. A card from a newer
MLRacecar is refused with a message to update.

### Refusing a model that doesn't fit

`SB3Agent.load(folder, spec)` compares the card with the observations the environment gives
now. If the digests differ, loading fails with every difference in plain words, for example:

```
models/first-lap: this model doesn't fit this environment:
  curvature: in the model, but turned off here
  rays: 19 values here, 15 in the model
  rays: count is 19 here, 15 in the model
```

Without a `spec`, nothing is checked; that's for looking at a model, not driving with it.

## Training on many cars at once

SB3 trains on its own vector-environment interface. `mlracecar.training.vec_env.SB3VecEnv` puts
the [many-cars environment](environment.md#many-cars-at-once) behind it, so training gets one
world of many cars instead of SB3's `DummyVecEnv` of separate environments:

```python
from gymnasium.vector import AutoresetMode
from stable_baselines3 import PPO

from mlracecar.env.batched import BatchedRacingEnv
from mlracecar.training.vec_env import SB3VecEnv

env = SB3VecEnv(BatchedRacingEnv("tracks/technical.json", 16, autoreset_mode=AutoresetMode.SAME_STEP))
model = PPO("MlpPolicy", env, device="cpu").learn(total_timesteps=100_000)
```

SB3 restarts a finished run in the same step, so the environment must use `SAME_STEP`. Each
car's info gets what SB3 looks for: `terminal_observation` when a run ends, and
`TimeLimit.truncated` when it was stopped (time limit, or stuck) rather than over, so learning
still counts on what would have followed. A test checks every transition against SB3's own
`DummyVecEnv` of single environments: they're identical.

On the dev machine, PPO (on the CPU) learned from about 5,700 car-steps a second on 16 cars
this way, against about 700 on one environment: 8 times faster.
