# The RL environment

> **In plain words:** This is the "game" the AI plays while it learns. Each step, the AI gives
> it a steering and a pedal value; it moves the car 1/20 of a second, and gives back what the AI
> now sees, its score for that step, and whether the run is over. It follows Gymnasium, the
> standard that RL libraries such as Stable-Baselines3 expect, so they can train on it as is.
> You can also watch it, in a window or as video frames.

`RacingEnv` (`mlracecar.env.racing`) ties together everything from the earlier pages: the
[world](race-rules.md), the [sensors](sensors.md), the [observation](observations.md), and the
[reward and the end-of-run rules](rewards.md). The ideas are in [RL fundamentals](rl-guide.md).

## Making one

```python
import gymnasium
import mlracecar.env  # registers MLRacecar-v0

env = gymnasium.make("MLRacecar-v0", track="tracks/technical.json")
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # a random driver
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

`gymnasium.make` takes:

| Argument | Meaning |
|----------|---------|
| `track` | A track file's path, or a `Track`. Required: there's no default track. |
| `config` | A `RacecarConfig` with every setting (the defaults if left out). Use `mlracecar.config.files.load_config` to read settings files and `--set`-style overrides. |
| `render_mode` | `"human"` (a window, in real time), `"rgb_array"` (`render()` returns a picture), or `None`. |

## What goes in and out

| | Space | Notes |
|-|-------|-------|
| Action | `Box(-1, 1, (2,), float32)` | `[steer, pedal]`: steer +1 is full left; pedal +1 full throttle, −1 full braking. Values outside are clipped; NaN is refused. |
| Observation | `Box(low, high, (31,), float32)` | The [observation](observations.md); the bounds come from its spec. |

`reset(seed=..., options=...)` starts a run. Options:

- `start`: `"grid"` (behind the start line) or `"random"` (anywhere on the lap, across the road,
  at rest). Without it, the `episode.start` setting decides; that's `grid` by default.
- `track`: change track for this run and the ones after it.

`step(action)` returns the observation, the reward, `terminated`, `truncated`, and `info`:

| `info` key | When | What |
|------------|------|------|
| `distance` | always | Metres driven along the track since the run began. |
| `speed` | always | m/s. |
| `laps` | always | Laps completed. |
| `best_lap` | always | The best valid lap time in seconds, or `None`. |
| `off_track` | always | Whether the car's centre is off the road. |
| `terms` | every step | This step's points per reward term. |
| `end_reason` | when the run ends | `off_track`, `out`, `time_limit`, or `stuck`. |
| `episode_terms` | when the run ends | The run's points per reward term: which terms its score came from. |

## The same seed, the same run

Everything random goes through the environment's own random generator, which `reset(seed=...)`
seeds. Two environments reset with the same seed and given the same actions produce exactly the
same observations, rewards, and endings; a test checks it. Starts on the grid involve no
randomness at all.

## Watching

`render_mode="human"` opens a window and shows every step as it happens, 20 a second, following
the car with its distance rays on. `"rgb_array"` draws the same picture offscreen (960 × 600)
and `render()` returns it, for recording videos (M4).

The environment itself never draws. Training runs without pygame, and the architecture keeps
the environment from importing the drawing code. Instead, `gymnasium.make` builds it through
`mlracecar.play.environment.make_racing_env`, which hands it a `RaceViewer`
(`mlracecar.render.viewer`) only when a render mode is asked for. pygame is only loaded then.

## Many cars at once

Training goes much faster with many runs at the same time. `BatchedRacingEnv`
(`mlracecar.env.batched`) is a Gymnasium **vector environment**: `num_envs` cars in one world,
each in its own run, moved, scored, and observed together. With 64 cars it's about 15 times
faster than 64 separate `RacingEnv`s (the [speed table](https://github.com/taofuuu/MLRacecar#speed)).

```python
env = gymnasium.make_vec("MLRacecar-v0", num_envs=64,
                         vectorization_mode="vector_entry_point", track="tracks/technical.json")
observations, infos = env.reset(seed=42)  # observations: (64, 31)
observations, rewards, terminated, truncated, infos = env.step(actions)  # actions: (64, 2)
```

The cars are ghosts that drive through each other, and each starts as if it were alone: on pole
position, or at a random place drawn from its own random generator (`reset(seed=s)` gives car
`i` the seed `s + i`, as Gymnasium's vector environments do). So every car's run is **exactly**
what a single `RacingEnv` with the same seed would give, bit for bit. A test checks it against
Gymnasium's `SyncVectorEnv` of separate environments.

When a car's run ends it starts again by itself, while the others carry on. `autoreset_mode`
(passed as a keyword to `make_vec`) says when:

| Mode | The step that ends a run returns | The car's next step |
|------|----------------------------------|---------------------|
| `NextStep` (default) | Its last observation, with `info["end_reason"]` and `info["episode_terms"]`. | Restarts it instead of driving: reward 0, not ended. |
| `SameStep` | The new run's first observation; the last one is in `info["final_obs"]`, the last info in `info["final_info"]`. | Drives on as usual. |

Stable-Baselines3 works the `SameStep` way; its adapter comes with the first trained agent (M4).

`infos` follows Gymnasium's vector convention: each key holds one entry per car, and
`infos["_key"]` says which cars have it (for example, only cars whose run just ended have an
`end_reason`). A best lap not yet set is `NaN` here, not `None`.

## Checks

- Gymnasium's own checker (`gymnasium.utils.env_checker.check_env`) passes, for every render
  mode.
- A random driver runs 10,000 steps without errors (`uv run pytest -m slow --no-cov`; a
  2,000-step version runs with every test run).
- Runs repeat exactly from a seed.
- `BatchedRacingEnv` gives exactly the transitions of separate environments with the same
  seeds, in both autoreset modes, from the grid and from random starts.
