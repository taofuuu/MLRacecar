# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- How the AI is scored (`mlracecar.env.rewards`): a reward made of named terms (progress
  along the lap, leaving the road, time, driving the wrong way, jerky controls, valid laps),
  each with a weight in the new `reward` settings section and its points reported separately.
  By default only progress (10 m = 1 point) and leaving the road (-10) count.
- When a training run ends (`mlracecar.env.episodes`): leaving the road ends it (terminated),
  whatever the race's own off-track rule; the 60-second time limit and being stuck for 5
  seconds stop it (truncated). Set in the new `episode` settings section.
- What the AI sees (`mlracecar.env.observations`): `ObservationBuilder` turns a snapshot into
  31 numbers per car, scaled to about -1..1: the distance rays, speed, heading (sine and
  cosine), offset from the middle of the road, yaw rate, steering angle, the previous action,
  and how much the road bends over 8 stretches of the next 150 m. Each input can be switched
  off in the new `observation` settings section. An `ObservationSpec` labels every value and
  has a digest that changes whenever the observation does, for tying trained models to it.
- Settings can now be on/off values, written `true` or `false`.
- An RL fundamentals guide (`docs/rl-guide.md`): the agent-environment loop, MDPs,
  observations, actions, rewards, episodes (termination vs. truncation), policy and value, PPO
  and why it clips, and what to watch while training, each tied to the code, with the choices
  the RL environment tickets will need and questions to check yourself.
- Distance sensors (`mlracecar.core.sensors`): rays fanned out from each car that measure how
  far the road's edges are, in metres and as a fraction of the range, exactly as if every piece
  of edge were tested but much faster. Set them up in the new `sensors` settings section (15
  rays across 180 degrees, reaching 100 m, by default), and press **4** in `racecar drive` to
  see them. Reading 15 rays for 64 cars takes about 1.4 ms on the dev machine.
- Simulation speed benchmarks: one world step for 1, 64, and 1,024 cars racing on the GP
  circuit. The README shows the baseline (1,024 cars run 15 to 20 times faster than real time
  on the dev machine), `scripts/benchmark_table.py` makes that table from the results, and CI
  measures again on every push to `main` and keeps the results as a download.
- Golden-trajectory regression tests: four recorded runs (a straight line, a slalom, a scripted
  lap, and four cars off the road and the wrong way) that every run must match to about a
  millionth, with a summary of what changed, when, and by how much when they don't. Record
  them again after an intended change with `scripts/update_golden.py`.
- `mlracecar.core.geometry`: vectorized 2D geometry (cross product, rotation, angle wrapping,
  segment intersection, raycasts, projection onto a polyline, self-intersection detection),
  property-tested with Hypothesis.
- `mlracecar.core.track.spline`: closed C2 cubic spline through control points (ADR-0010) with
  even arc-length resampling, unit tangents and normals, and signed curvature.
- `mlracecar.core.track.model.Track`: road widths blended smoothly between control points, left
  and right edges, evenly spaced checkpoints with the start/finish line at control point 0,
  poses at any distance along the track, and a staggered two-wide starting grid.
- `mlracecar.core.track.validation.validate`: checks a track and reports every problem with a
  severity, a stable code, a plain-language message, and a location (control point, stretch of
  road, or spot) for the editor to highlight. Rules: too few points, invalid or coinciding
  points, non-positive widths, too narrow, too short, edges folding in tight bends, inside
  edges coming to a sharp point (warning), bends too tight to steer (warning), the track
  crossing itself, and parts of the road overlapping. Each bend, and each run of too-narrow
  points, is reported once.
- Track files (`mlracecar.io.track_file`): versioned JSON (format version 1) with field-level
  error messages, automatic upgrades from older versions, refusal of newer ones, atomic saving,
  one control point per line, and a published JSON Schema (`docs/schemas/`).
- Sample tracks in `tracks/`: an oval, a 1.1 km technical circuit with 14 corners for
  practising car control, and a 3.5 km GP circuit, all free of track-check issues.
- `racecar check <file>`: reads a track file and runs the track checks on it.
- Settings (`mlracecar.config`, ADR-0008): pydantic models for the car and the simulation
  timing, read from YAML files and `--set section.key=value` in layers (defaults < files <
  `--set`), converted to the core's `VehicleParams` and `Timing`. Errors list every invalid
  setting with the file it came from and suggest the nearest name for a typo.
  `configs/default.yaml` lists every setting with its default; the default car is a light
  race car (0-100 km/h in about 3 s, cornering at up to 2 g, braking at almost 3 g).
- `racecar config [files] --set key=value`: shows the settings a combination of files and
  overrides produces, or what is wrong with them.
- `racecar drive <track>`: drive a car round a track with the keyboard (arrow keys or WASD),
  with lap times, R to restart, C for the camera, 1-3 for debug overlays, P to pause, and the
  best lap of the session in the title bar. It takes the same `--config` and `--set` options
  as `racecar config`, and refuses tracks with errors. The race runs in real time while the
  window draws 60 frames a second, blending the car between decisions. The follow camera
  looks ahead of the car and zooms out with speed to keep 2.5 seconds of road in view,
  easing into place (about half a second) instead of jumping when the car brakes or steers.
- The `Agent` protocol (`mlracecar.agents.base`) and `KeyboardAgent`, which turns held keys
  into smooth steering and a pedal.
- The race renderer (`mlracecar.render.race.RaceRenderer`): draws snapshots only, with the
  track and its kerbs, the cars, a HUD (speed, lap, current, last and best lap times,
  off-track and wrong-way warnings), three cameras (follow, overview, free), and toggleable
  debug overlays (centerline, checkpoints, velocity, sensor rays). It draws into a window or
  offscreen as RGB arrays, headless too, at 6-9 ms a frame with 16 cars. `Snapshot` and
  `RaceState` moved to data-only modules (`core.snapshot`, `core.race.state`).
- Race rules, part 2: a car whose centre leaves the road is off track (`OffTrack` event), and
  the `race.off_track` setting says what happens: nothing, the grass slows it down (default,
  `race.grass_slowdown` m/s per second), it is put back on the road, or its run ends until it
  is reset. A car moving backwards along the track is going the wrong way (`WrongWay` event).
- Race rules (`mlracecar.core.race`), part 1: each car's position along the lap, distance
  driven, offset from the middle of the road and heading error; checkpoints that must be
  crossed in order (backwards undoes, missing one invalidates the lap); laps, lap and sector
  times, and best and latest lap; `LapCompleted` events in each snapshot. See the new Race
  rules page.
- The simulation world (`mlracecar.core.world.World`): N cars on a track, stepped one driver
  decision at a time (each action held for `action_repeat` physics steps), with frozen
  `Snapshot`s whose arrays are read-only. `reset` puts chosen cars back on their own grid
  spots or at random places on the road, at rest, without touching the others. The same seed
  and actions give exactly the same race. One decision for 1024 cars takes about 0.6 ms.
- Car physics (`mlracecar.core.vehicle`, ADR-0014): `VehicleState` for N cars at once, and
  the `KinematicBicycle` model behind a `DynamicsModel` protocol. Steering turns the wheels
  at a limited rate; the engine's push fades with speed once its power runs out; brakes, air
  drag and rolling resistance slow the car, which has no reverse gear. A new `grip` setting
  limits cornering, so a car too fast for a bend runs wide. The default car does 0-100 km/h
  in 3.2 s and stops from 100 km/h in 13.3 m. See the new Car physics page.
- `mlracecar.editor.draft.TrackDraft`: the track editor's headless model. Immutable drafts
  (ADR-0011) with append, insert-into-stretch, move, delete, width, reverse-direction, and
  set-start edits, cached track and validation per draft, and conversion to and from track files.
- The track editor window, `racecar edit [file]`: click to draw a track and watch the road
  appear, click on the road to insert a point there (shift+click always adds after the last
  point), drag to move, right-click to delete, shift+wheel or `[`/`]`
  for road width, pan and zoom, grid snap (G), fit (F), reverse (R), start line (S), and a help
  panel (H). Opens existing track files; saving comes next.
- Saving and live track checks in the editor: Ctrl+S saves (asking for a file name the first
  time; untitled tracks are named after their file), Ctrl+Shift+S saves as (asking before
  replacing another file), F2 renames, and closing with unsaved changes asks first (the title
  bar shows `*`). Track problems are marked on the map while you draw (red errors, orange
  warnings) and listed in a corner; pointing at a marker shows the full message, and the status
  bar says whether the track can be raced. Tracks with errors can still be saved, with a
  warning. The checks run on a worker thread (ADR-0013), so dragging stays at 60 fps on a
  3.5 km track.
- Undo and redo in the editor: Ctrl+Z undoes, Ctrl+Y or Ctrl+Shift+Z redoes, up to 500 steps.
  One step is one whole action: a click, a drag, or a run of width changes at one point.
  Undoing back to the saved track clears the unsaved-changes star. The editor now saves
  positions and widths to the centimetre, so track files stay readable.
- `mlracecar.render`: an immutable `Camera` (metres to pixels, pan, zoom around the cursor,
  fit, grid spacing) and pygame drawing of the grid and the track (road, edges, start line,
  direction arrow) that fills only what's on screen, for a few milliseconds per frame at any
  zoom.
- The `render` extra now installs pygame-ce (ADR-0012). An architecture rule keeps pygame out
  of everything that must run headless.
- A realistic 3.5 km GP-style test circuit (`tests/circuits.py`) that validation is checked
  against, both as designed and pushed past its limits.
- `mlracecar.core.geometry.polyline_crossings`, and a bounding-box broad phase for all crossing
  searches: about 46x faster on a 1.5 km track (validation takes ~33 ms there).
- Project vision, architecture, ADRs 0001–0009, roadmap, and contribution workflow.
- Backlog definition and an idempotent GitHub seeding script.
- Plain-language summaries on every planning document, plus a glossary.
- Python package scaffold (`src/mlracecar`, uv, Python 3.12) with a `racecar` CLI
  (`racecar --version`).
- Code-quality gates: ruff lint and format, strict mypy, and pre-commit hooks.
- Test framework: pytest with coverage (80% minimum), Hypothesis, pytest-benchmark, and opt-in
  `slow` / `gpu` / `benchmark` markers.
- Continuous integration on GitHub Actions: all pre-commit hooks, plus tests on Ubuntu and
  Windows with a coverage summary. `main` is protected: merging requires a PR with green CI.
- Architecture layer packages (`core`, `config`, `io`, `env`, `render`, `agents`, `training`,
  `editor`) with automated boundary checks: import-linter for the layer order and an
  allow-list test keeping `core` to the standard library and NumPy.
- Issue forms (feature, bug, research spike) that label new tickets and add them to the board,
  and a pull request template with the Definition of Done.
- Documentation website (MkDocs Material) with Mermaid diagrams and an API reference generated
  from docstrings; built in strict mode on every PR and published to GitHub Pages from `main`.

### Fixed

- A car's heading error and the road width used for the off-track check now change smoothly
  along the road. They stepped at every 0.5 m centerline sample, so a car almost equally near
  two samples could get either value, and Windows and Linux could disagree.
- `wrap_angle` returns angles that are already in range unchanged. Before, it could change
  them by a rounding error, which would have made a car driving straight drift off course.
- Control points closer than a micrometre now count as on top of each other: the track checks
  report them, and the track code refuses them, instead of dividing by zero.
- Track files saved as UTF-16 (what Windows PowerShell 5.1's `>` and `Out-File` write) or as
  UTF-8 with a byte-order mark now open. A file that isn't text at all gets the usual
  "Can't open the track." message from `racecar check` and `racecar edit`, instead of a crash.
