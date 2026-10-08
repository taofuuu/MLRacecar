# ADR-0015: Replays as snapshot streams in NumPy's npz

- **Status:** Accepted
- **Date:** 2026-10-08
- **Related:** [#37](https://github.com/taofuuu/MLRacecar/issues/37) (replays),
  [#38](https://github.com/taofuuu/MLRacecar/issues/38) (video export), ghosts (M5), the web
  demo (M6), [ADR-0003](0003-layered-architecture-pure-core.md),
  [ADR-0005](0005-batched-multi-car-simulation.md)

> **In plain words:** A replay file keeps a picture of the race (a snapshot) for every driver
> decision, 20 a second, plus the track and the settings, so playing it back shows exactly what
> happened, even after the code changes. The file is NumPy's own compressed format (npz): one
> column of numbers per thing recorded. A minute of one car is about 110 kB, a lap about 57 kB.

## Context

Replays feed several things: watching a run again (`racecar replay`), videos for the README
(#38), ghost cars to race against (M5), and the web demo (M6). They need to be:

- **exact:** a replay of an evaluation run must match the run that was scored, bit for bit;
- **lasting:** still playable after the simulation, the settings, or the track file change;
- **compact:** under 1 MB a lap (the ticket's limit), and fine for long races with many cars;
- **versioned**, and **safe** to open: reading a file must never run code from it.

The race is already a stream of immutable snapshots (architecture 4.4), each a handful of
arrays with one entry per car (struct-of-arrays, ADR-0005), and the renderer draws from
snapshots alone (ADR-0003).

## Options considered

**What to record:**

1. **Inputs:** the seed and every action, replayed by simulating again. Tiny files, but a replay
   breaks whenever the physics or the rules change, and it isn't even exact today: a neural
   network rounds its last digits differently acting for one car than for several (about 1e-7),
   so driving an evaluation run again alone can drift from the run that was scored.
2. **Snapshots:** every snapshot, as it was. Bigger, but exact, independent of the code that
   made it, and played back without simulating.

**How to store snapshots** (measured on a minute of one car, 1,201 snapshots):

1. **npz:** a zip of NumPy arrays, compressed; one array per snapshot field with the snapshots
   along the first axis, and the rest (track, settings, events, information) as a JSON text.
   107 kB. No new dependency; arrays load whole, fast, so scrubbing is instant; scales to many
   cars. A browser needs a small reader (unzip and parse `.npy`), or reads it directly if the
   web demo runs Python (Pyodide).
2. **MessagePack:** a compact binary format read by every language, browsers included. About
   the same size; a new dependency, and NumPy arrays need custom packing.
3. **Gzipped JSON:** plain text, readable once unzipped, opened by a browser with no library.
   132 kB, slower to parse, and it grows badly with long multi-car races.

## Decision

A replay records **snapshots**, in an **npz** file (`mlracecar.io.replay`): `tick`, `time`, and
one array per `VehicleState` and `RaceState` field (`cars.x`, `race.laps`, ...) with the
snapshots along the first axis, and `meta`, a JSON text holding the format name and version, the
track file, every setting, the race events, and free-form information (who drove, how it went).
Files are read with `allow_pickle=False`, and a newer format version is refused by name.

## Consequences

- **Positive:** replays are exact and outlive code changes; one lap is about 57 kB; no new
  dependency; any snapshot is one array lookup away, so scrubbing and video export are simple.
  Evaluation records the scored run itself (`drive_test_runs(on_step=...)`), not a re-drive.
- **Negative / costs:** files are larger than input recordings would be; a browser needs an npz
  reader unless the demo runs Python; adding a snapshot field means a new format version (or
  reading old files without it).
- **Follow-ups:** video and GIF export from replays (#38); recording your own drives, with
  ghost laps (#46, M5); the web demo decides how it reads replays (M6).
