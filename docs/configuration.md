# Settings

> **In plain words:** The numbers the simulation runs on, such as the car's weight, power, and
> steering, and how fast simulated time moves, live in small text files. Every value is checked
> when the files are read, so a typo gets a clear error straight away. To change a setting, you
> only write down the ones you want different; everything else keeps its default.

Settings files are YAML. The reasoning behind the design is in
[ADR-0008](adr/0008-typed-configuration.md).

## Every setting and its default

[`configs/default.yaml`](https://github.com/taofuuu/MLRacecar/blob/main/configs/default.yaml)
lists every setting with its default value and a one-line explanation. It is written from the
code, and a test fails if the two ever disagree, so it is always up to date. An extract:

```yaml
# The car: its size, steering, engine and brakes, and what slows it down.
vehicle:
  length: 4.5                # Length of the body, in metres.
  mass: 1300                 # The car's mass, in kilograms.
  max_steer: 30              # How far the front wheels turn at full lock, in degrees.
  max_power: 200             # Engine power in kilowatts; it limits the push at high speed.

# How simulated time moves forward.
simulation:
  physics_hz: 120   # Physics steps per simulated second.
  action_repeat: 6  # Physics steps per driver decision (120 / 6 = 20 decisions a second).
```

The default car is a hot sporty car: 0–100 km/h in about 3 seconds, about 300 km/h flat out
(see [Car physics](vehicle-model.md)). Angles are in degrees and power
in kilowatts, because those are easy to picture; the simulation converts them to radians and
watts.

A third section, `race`, holds the race rules: what happens when a car leaves the road
(`off_track`: `none`, `slowdown`, `reset`, or `terminate`) and how hard the grass slows it
(`grass_slowdown`). See [Race rules](race-rules.md).

## Changing settings

Write a file with only the settings you want to change, under their section:

```yaml
# heavy-car.yaml
vehicle:
  mass: 1800
```

Or change one setting on the command line with `--set section.key=value`. Settings are built
in layers, and each layer changes only what it mentions:

| Layer | Example | Wins over |
|-------|---------|-----------|
| 1. Defaults | the code (`configs/default.yaml` shows them) | nothing |
| 2. Settings files, in the order given | `heavy-car.yaml` | the defaults and earlier files |
| 3. `--set`, in the order given | `--set vehicle.mass=1500` | everything else |

To see the settings a combination of layers produces, use `racecar config`:

```bash
uv run racecar config heavy-car.yaml --set vehicle.max_steer=25
```

It prints the final settings in the same form as `configs/default.yaml`, so its output is
itself a settings file. Later, `racecar drive` and `racecar train` will take the same files and
`--set` options, and every training run will save the final settings it used.

## Checks

Reading settings rejects anything that can't be right, and lists every problem at once with the
file (or `--set`) it came from:

```text
Invalid settings:
  vehicle.width: must be a number, got "wide" (from heavy-car.yaml)
  vehicle.wheelbase: must be shorter than the car's length (2.0 m), got 2.7
  vehicle.max_stear: unknown setting; did you mean max_steer? (from --set)
```

The wheelbase problem names no file: it's the default wheelbase, which doesn't fit the 2 m car
that `heavy-car.yaml` asked for.

- **Unknown settings are errors**, so a typo can't be silently ignored.
- **Types are strict:** `"1300"` (text) isn't a number, and `60.0` isn't a whole number where
  one is needed. Whole numbers are fine where decimals are expected.
- **Numbers must be finite** and within their limits, for example a mass above 0 and a steering
  lock below 90°. Some settings are checked against each other: the wheelbase must be shorter
  than the car.
- **A setting written twice in one file is an error.** Plain YAML would quietly keep the second.
- **Numbers like `1e3` are numbers.** Plain YAML 1.1 would read them as text.

## Adding a setting

For developers: settings are pydantic models in `mlracecar.config.models`.

1. Add a field with a default, its limits (for example `Field(gt=0)`), and a one-line
   docstring. The docstring becomes the setting's comment in the files MLRacecar writes.
2. Pass it on in the section's conversion method (for example `VehicleConfig.to_params`). The
   simulation only ever receives these plain frozen dataclasses, never pydantic models.
3. Rewrite the defaults file: `uv run python scripts/export_default_config.py`.

New parts of MLRacecar add their own sections: race rules, the RL environment, and training.
