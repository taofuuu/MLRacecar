# Training

> **In plain words:** `racecar train` teaches an AI to drive. It lets 16 cars practise on a
> track at once, and every so often it tests the AI on 10 runs, keeping the best version so far.
> Everything about the run goes into one folder: the exact settings, which code and library
> versions, the seeds, the saved AIs, and every test result. Run the same settings with the
> same seed again on the CPU and you get exactly the same AI. Press Ctrl+C to stop; the AI is
> saved first, and you can carry on later. While it trains, TensorBoard (a web page) draws
> charts of how it's going, with a video of the AI driving every so often.

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
| | `video_every` | 200,000 | Film the test run from the grid every this many car-steps, and at the end, for TensorBoard. 0: no videos. |
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

## Watching it learn in TensorBoard

```bash
uv run tensorboard --logdir runs
```

Then open <http://localhost:6006>. Each run writes its charts into its own `tensorboard/`
folder as it trains, so the page follows it live (it refreshes every 30 s, or press its reload
button), and every run under `runs/` is a line on the same charts, to compare.

**Practice and tests.** *Practice runs* are the cars' runs while the AI learns. It still acts a
little randomly, to explore, so they're noisier and a bit worse than the tests, where it always
acts on its best guess. The practice numbers are over the latest 100 runs that ended, written
after every update; the test numbers come from each test.

| Charts | What they show |
|--------|----------------|
| `practice/`, `test/` | `score` (mean reward), `distance` (metres along the lap), `average_speed` (m/s), `lap_rate` (the share of runs with a valid lap), and `best_lap` (seconds, once there is one). |
| `practice_ends/`, `test_ends/` | The share of runs that ended each way: `off_track`, `out`, `time_limit`, `stuck`. |
| `practice_reward/`, `test_reward/` | Each reward term's mean points per run: what the score is made of ([Rewards and episodes](rewards.md)). |
| `test/grid_*` | The run from the grid: `grid_score`, `grid_distance`, and `grid_best_lap`. |
| `rollout/` | Stable-Baselines3's practice numbers: `ep_rew_mean` (the same as `practice/score`) and `ep_len_mean` (steps per run). |
| `train/` | How PPO's learning is going: its losses, `approx_kl`, `clip_fraction`, `entropy_loss`, `explained_variance`, and `std` ([RL fundamentals](rl-guide.md) §9). |
| `time/fps` | Car-steps a second since the run (or the resume) started, tests and videos included. |

Stable-Baselines3 writes an update's `train/` numbers after the next stretch of practice, so
they show one update late, and the last update's aren't written.

**Videos.** Every `video_every` steps (200,000 by default) and at the end, the test run from the
grid is filmed and shown in TensorBoard's **Images** tab as `test/grid_run`, with a slider to
pick the step. It's the same run as **grid** in the line above, so you can watch what the
numbers say: a rising score with a car cutting across the grass is reward hacking. Videos are
half size (480 × 300) at 10 pictures a second: about 4 MB, and 6–10 s to make, per minute of
racing. They need pygame (`uv sync --extra render`, part of the developer setup); without it,
`racecar train` says there are no videos and trains anyway.

**Other trackers.** Everything goes through a `Tracker` (`mlracecar.training.tracking`) with three
methods: `scalars(step, values)`, `video(step, name, frames, fps)`, and `close()`.
`TensorBoardTracker` is the default; to send everything somewhere else, such as Weights &
Biases, pass another to `TrainingRun.train(tracker=...)`.

## The run folder

```
runs/2026-10-08_153012_technical-seed0/
├── config.yaml              every setting, as resolved (defaults < files < --set)
├── meta.json                status, git commit, versions, seeds, track, hardware, timings
├── checkpoints/
│   ├── step_000100000/      the agent every training.checkpoint_every steps
│   ├── best/                the agent that scored best in testing
│   └── last/                the latest agent: what --resume carries on from
├── eval/evaluations.jsonl   one line per test
└── tensorboard/             charts and videos for TensorBoard
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
`lap_rate`, `best_lap`, `end_reasons` (how many runs left the road, ran out of time, ...), and
`terms` (each reward term's mean points per run). It also has `grid`, the run from the starting
grid, with its own `terms`.

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
