# Replays

> **In plain words:** A replay is a recording of a race that you can watch again: pause it,
> skip around, and slow it down or speed it up. `racecar eval --record replays/` saves every
> run the AI drives in an evaluation, and `racecar replay` plays one, or saves it as a GIF or an
> MP4 video to share. A replay keeps a picture of the race for every driver decision, so it shows
> exactly what happened, even after the code or the track changes. A one-minute run is about
> 110 kB.

## Recording

Evaluation saves every run it scores when given a folder ([Evaluation](evaluation.md)):

```bash
uv run racecar eval --model runs/<run>/checkpoints/best --record replays
```

Each run becomes one file, named after its track, agent, and number: `technical-A-01.npz`,
`technical-A-02.npz`, ... Each replay is the run that was scored, recorded as it was driven, so
its laps, times off the road, and distance are the report's.

Recording your own drives comes with ghost laps ([#46](https://github.com/taofuuu/MLRacecar/issues/46), M5).

## Watching

```bash
uv run racecar replay replays/technical-A-01.npz
```

The window works like [`racecar drive`](https://github.com/taofuuu/MLRacecar#try-it),
with a timeline along the bottom showing where the replay is up to:

| Key or mouse | What it does |
|--------------|--------------|
| Space | Play or pause. At the end, it starts again. |
| Left / Right | Back or on one second. |
| Up / Down | Faster or slower: x0.25, x0.5, x1, x2, x4. |
| Home / End | The start or the end. |
| Click or drag the timeline | Go anywhere. |
| C | The camera: following the car, the whole track, or free. |
| 1-4 | Overlays: centerline, checkpoints, velocity, distance rays. |
| Wheel, drag elsewhere | Zoom and move the view. |
| Esc | Close. |

The car moves smoothly at any speed: between two recorded moments it's drawn part of the way
from one to the next, as in the driving window. It needs pygame (the `render` extra).

## Videos

`--export` saves a replay as a video instead of opening the window: a GIF or an MP4, by the
file's extension.

```bash
uv run racecar replay replays/technical-A-01.npz --export lap.gif --lap 1 --camera overview
uv run racecar replay replays/technical-A-01.npz --export clip.mp4 --from 10 --to 25 --rays
```

| Option | Default | Meaning |
|--------|--------:|---------|
| `--export FILE` | | A `.gif` or `.mp4` file to save it as. |
| `--camera` | `follow` | `follow` (the chase camera) or `overview` (the whole track). |
| `--rays` | off | Show the distance rays the car drives by. |
| `--from`, `--to` | the whole replay | Where to start and stop, in seconds. |
| `--lap N` | | Just the Nth valid lap, from the line to the line. |
| `--fps` | 25 | Pictures a second. A GIF's timing is in hundredths of a second, so 10, 20, 25, or 50 play exactly. |
| `--size` | `960x600` | Width x height in pixels (even, for an MP4). |

`--camera`, `--rays`, `--from`, and `--lap` work when watching too: the window opens that way.

GIFs need nothing more; MP4s need the `video` extra (`uv sync --extra video`, about 30 MB:
the ffmpeg encoder). An MP4 is about a quarter of the size of the same GIF, in full colour:
better for sharing on social media. A GIF plays anywhere, including in a README. One lap takes
5 to 10 seconds to make. The follow camera's GIFs are bigger, as every picture moves (a lap at
640 x 400: about 6 MB, against 1.8 MB for the overview).

### The README's GIF

The GIF at the top of the README is the AI's first clean lap, made from a replay kept in the
repository (`docs/media/hero-lap.npz`) by one command:

```bash
uv run racecar replay docs/media/hero-lap.npz --export docs/media/hero.gif --lap 1 --camera overview --size 640x400 --rays
```

A test checks that the replay still reads and the command still works.

## The file

A replay is NumPy's `.npz`: a compressed zip of arrays (decided in
[ADR-0015](adr/0015-replays-as-npz-snapshot-streams.md)).

| Array | Shape | What |
|-------|-------|------|
| `tick`, `time` | `(snapshots,)` | Physics steps and seconds since the race began. |
| `cars.x`, `cars.y`, `cars.yaw`, `cars.vx`, `cars.vy`, `cars.yaw_rate`, `cars.steer` | `(snapshots, cars)` | Each car's position and motion ([Car physics](vehicle-model.md)). |
| `race.*` (`distance`, `laps`, `last_lap`, `off_track`, ...) | `(snapshots, cars)` | Each car's race ([Race rules](race-rules.md)); `race.splits` has a third axis, one entry per sector after the first. |
| `meta` | a JSON text | `format` (`mlracecar-replay`), `schema_version` (1), the `track` file, every `settings`, the race `events` (each with its `step`), and `info`. |

For an evaluation run, `info` says who drove (`agent`: label, path, model fingerprint, run,
steps), on which track, which run it was with its `seed` and `start`, and its `result`, as in
the report.

Reading a replay never runs code from the file (it holds only numbers and JSON). A replay from a
newer format version is refused with its version number, rather than misread.

## Why record the race, not just the actions?

Recording the seed and every action, and simulating again, would make much smaller files. But
such a replay breaks whenever the physics or the race rules change, and it wouldn't even be exact
today: the AI's neural network rounds its last digits slightly differently when it acts for one
car than for twenty at once, so driving a scored run again on its own can drift from it. A
replay of snapshots is the race as it happened, whatever changes later.
