# Evaluation

> **In plain words:** `racecar eval` tests saved AIs and writes a report card. Each AI drives
> 20 runs on each track, from random places on the lap, without learning, and the report says
> how often it got through a run without leaving the road (the **completion rate**), its laps
> and lap times, and its speed. Give it two AIs and it tests both on exactly the same runs and
> puts them side by side. The same AIs and seed always give exactly the same report.

`racecar eval` needs the training libraries (`uv sync --extra train`), like
[`racecar train`](training.md). The agents it scores are the saved agents a training run keeps
([Trained agents](agents.md)).

## Scoring an agent

```bash
uv run racecar eval --model runs/2026-10-08_174301_tb-check/checkpoints/best
uv run racecar eval --model runs/.../checkpoints/best --tracks "tracks/*.json" --episodes 50
uv run racecar eval --model runs/.../checkpoints/best --json best.json --markdown best.md
```

| Option | Default | Meaning |
|--------|--------:|---------|
| `--model FOLDER` | (needed) | A saved agent's folder. Repeat to compare several. |
| `--tracks FILES` | the track each agent trained on | Track files, or a pattern in quotes (`"tracks/*.json"`). Repeat for more. |
| `--episodes N` | 20 | Runs per agent and track. |
| `--seed N` | 0 | Sets the start places: the same seed, the same places. |
| `--start` | `random` | `random` places on the lap, or `grid`. |
| `--set KEY=VALUE` | | Change one of the agents' settings for this evaluation. |
| `--json FILE`, `--markdown FILE` | | Also save the report. |
| `--record FOLDER` | | Save every run as a replay, to watch with `racecar replay` ([Replays](replays.md)). |

The report is printed as Markdown; what's being driven goes to the error stream, so
`racecar eval ... > report.md` saves just the report. A track with errors, a folder without a
saved agent, or a setting that can't be used stops it with a one-line reason and status 1.

## Comparing agents side by side

Give several `--model`: they're labelled A, B, C, ... in order, and every agent drives the same
runs, from the same places, on the same tracks. Testing an older agent again costs seconds and
gives the same numbers every time, so there's no need to keep old reports to compare against.

```bash
uv run racecar eval --model runs/2026-10-08_174301_tb-check/checkpoints/best \
                    --model runs/2026-10-08_174301_tb-check/checkpoints/step_000100000
```

```
| Agent | Model                                                      | Algorithm | Run      |   Steps |
| ----- | ---------------------------------------------------------- | --------- | -------- | ------: |
| A     | runs/2026-10-08_174301_tb-check/checkpoints/best           | PPO       | tb-check | 301,056 |
| B     | runs/2026-10-08_174301_tb-check/checkpoints/step_000100000 | PPO       | tb-check | 100,000 |

## Technical Circuit (tracks/technical.json)

|                 |            A |         B |
| --------------- | -----------: | --------: |
| Completion rate | 100% (20/20) | 0% (0/20) |
| Laps            |           16 |         0 |
| Mean lap        |      30.85 s |         - |
| Best lap        |      30.33 s |         - |
| Left the road   |            0 |        20 |
| Average speed   |     33.2 m/s |  25.7 m/s |
| Distance        |      1,991 m |     228 m |
| Score           |       199.09 |     12.82 |
```

That's the agent from a 300,000-step run against the same run's agent at 100,000 steps: by
the end it got through every run without leaving the road.

## What the report says

| Row | Meaning |
|-----|---------|
| Completion rate | The share of **clean runs**: runs that lasted until the time limit without the car ever leaving the road (or getting stuck, or being taken out). |
| Laps | Valid laps in all the runs together. A run from a random place first has to reach the start line, so a 60 s run of 30 s laps often has one lap, not two. |
| Mean lap, Best lap | The mean and the best of every valid lap's time, or `-` without a lap. |
| Left the road | How many times a car left the road, in all the runs together. When leaving the road ends the run (the default, `episode.end_off_track`), that's the runs that did. |
| Average speed | Metres along the track per second, averaged over the runs. |
| Distance | Metres along the track per run, on average. |
| Score | The mean reward per run, as in training. It depends on each agent's reward settings. |

The completion rate counts every crash, wherever the run started and however long the laps
take, which is why it's the headline number.

## Start places

The runs start from random places on the lap (anywhere along it, across the road, at rest),
set by `--seed`: the same seed gives the same places, for every agent and every track. They are
**not** the places training tests on: training keeps the agent that scored best on its test
places, and testing it again on those would flatter it. `--start grid` starts every run on the
grid instead; an agent always drives the same way from the same place, so all its runs are the
same.

## Settings

Each agent drives with the settings it was trained with, from its model card: the same car,
sensors, observations, rules, and time limit. `--set` changes them for this evaluation, for
example a longer run with `--set episode.time_limit=120`, and the report says so. A change to
what the agent sees (its observations or sensors) is refused, naming the difference, because the
network can't use inputs it wasn't trained on.

When agents were trained with different racing settings, the report lists the differences
(for example `episode.time_limit`: A 60.0, B 90.0), since they affect the numbers. How they
learned (the `training` and `ppo` sections) is left out.

## The JSON report

`--json` saves everything the Markdown shows and more:

- `schema_version`, `episodes`, `seed`, `start`, `seeds` (each run's seed), and `overrides`;
- `agents`: each agent's `label`, `path`, `model_sha256` (a fingerprint of `model.zip`),
  `algorithm`, `run`, `steps`, and every setting it drove with;
- `tracks`: each track's `path`, `name`, and `sha256`, and per agent a `summary` and every run
  (its reward, distance, lap times, times off the road, how it ended, how long it lasted, and
  its reward terms).

A report holds no dates or timings, so the same agents, tracks, settings, and seed give the
same report, byte for byte: a test checks it, and two reports can be compared with any diff
tool.
