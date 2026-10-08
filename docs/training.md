# Training

> **In plain words:** `racecar train` teaches an AI to drive. It lets 16 cars practise on a
> track at once, and every so often it tests the AI on 10 runs, keeping the best version so far.
> Everything about the run goes into one folder: the exact settings, which code and library
> versions, the seeds, the saved AIs, and every test result. Run the same settings with the
> same seed again on the CPU and you get exactly the same AI. Press Ctrl+C to stop; the AI is
> saved first, and you can carry on later.

Training needs the training libraries: `uv sync --extra train` (see
[CONTRIBUTING](https://github.com/taofuuu/MLRacecar/blob/main/CONTRIBUTING.md)). The ideas
behind it are in [RL fundamentals](rl-guide.md); the agent it trains is described in
[Trained agents](agents.md).

## Starting a run

```bash
uv run racecar train                                   # the default settings
uv run racecar train configs/smoke.yaml                # a tiny run: checks everything works
uv run racecar train --set training.steps=200000 --name short-test
```

Settings files and `--set` work as for every other command ([Settings](configuration.md)). The
two sections for training:

| Section | Setting | Default | Meaning |
|---------|---------|--------:|---------|
| `training` | `track` | `tracks/technical.json` | The track file to train on. |
| | `steps` | 1,000,000 | How long to train, in car-steps (16 cars driving one step = 16). PPO learns in whole updates, so a run can go a little past it (to the next 2,048 by default). |
| | `cars` | 16 | Cars practising at once, in one world. |
| | `seed` | 0 | Seeds everything random. |
| | `device` | `cpu` | Where the network learns: `cpu`, `cuda` (the GPU), or `auto`. The CPU is faster for networks this small. |
| | `checkpoint_every` | 100,000 | Save the AI every this many car-steps, and at the end. |
| | `eval_every` | 50,000 | Test it every this many car-steps, and at the end. |
| | `eval_runs` | 10 | Test runs per test. |
| `ppo` | `learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `entropy_coef`, `epochs` | SB3's usual values | How PPO learns ([RL fundamentals](rl-guide.md) §8). |
| | `steps_per_car` | 128 | Steps each car drives between updates. 16 cars × 128 = 2,048 steps of experience per update. |
| | `batch_size` | 256 | Steps per learning step. It must divide the experience per update; a run that doesn't is refused before it starts. |
| | `layers`, `layer_size` | 2, 64 | The networks' hidden layers. |

The track is checked first: a track with errors is refused, as in `racecar drive`.

While it runs, a line after each test says how it's going:

```
step   50,000/1,000,000  score    41.20  (best)     412 m  laps 0  best lap -  grid 388 m
```

The **score** is the test runs' mean reward (with the default reward, a tenth of the metres
driven along the lap, minus 10 for each time the car left the road). **laps** counts valid laps
in the test runs, and **grid** is how far one run from the starting grid got.

## The run folder

```
runs/2026-10-08_153012_technical-seed0/
├── config.yaml              every setting, as resolved (defaults < files < --set)
├── meta.json                status, git commit, versions, seeds, track, hardware, timings
├── checkpoints/
│   ├── step_000100000/      the agent every training.checkpoint_every steps
│   ├── best/                the agent that scored best in testing
│   └── last/                the latest agent: what --resume carries on from
└── eval/evaluations.jsonl   one line per test
```

`runs/` is not committed: a run can be made again from its `config.yaml` and the commit in
`meta.json`. Each agent folder is an [`SB3Agent`](agents.md): the model and its model card.

**`meta.json`** holds:

- `status`: `created`, `running`, `finished`, or `interrupted`;
- `git` (commit and uncommitted changes), `versions`, and `hardware` (processor, threads,
  device, GPU);
- `track`: the file's path and a SHA-256 of its contents;
- `seeds`: the run's seed, each car's (`seed + i`), and the test runs';
- `steps`, `best` (the best test's step and score), and one entry per `sessions` (each start
  or resume: from which step to which, how long, and how many steps a second).

**`eval/evaluations.jsonl`** has one JSON object per test: the step, the time since the
session started, and the test runs' `score`, mean `distance` and `average_speed`, total `laps`,
`best_lap`, and `end_reasons` (how many runs left the road, ran out of time, ...). It also has
`grid`, the run from the starting grid.

## Testing during training

Every `eval_every` steps, and at the end, the agent drives `eval_runs` runs without learning,
always acting on its best guess (deterministic). They start from random places on the lap that
are the **same at every test** (their seeds are in `meta.json`), so scores are comparable from
one test to the next, and every corner gets tested. All the test runs are driven at once in
one world, so ten take about as long as one. The best-scoring agent is copied to
`checkpoints/best`.

## The same run again

On the CPU, the same settings and seed give exactly the same run: the same network, to the
last bit, and the same test results. A test trains twice and compares; the smoke run does the
same in CI. This holds on one computer with the same library versions; a different processor
or PyTorch version can round differently and drift.

## Stopping and carrying on

**Ctrl+C** stops the run cleanly: the agent is saved to `checkpoints/last`, the run is marked
`interrupted`, and the last line says how to carry on:

```bash
uv run racecar train --resume runs/2026-10-08_153012_technical-seed0
```

It carries on with the run's own settings, until `training.steps`. The cars start fresh runs,
so a resumed run keeps learning but won't match an uninterrupted one exactly. If the track file
has changed since the run started, carrying on is refused; start a new run instead. A finished
run can't be resumed either.

## The smoke run

`configs/smoke.yaml` trains for about 2,000 car-steps on 8 cars, testing and saving twice: far
too little to learn to drive, but it goes through every part of the pipeline in a few seconds.
CI runs it twice on every pull request (`uv run pytest -m slow --no-cov`). It must take under a
minute, and both runs must give the same test results.
