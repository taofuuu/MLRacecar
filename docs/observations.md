# Observations

> **In plain words:** This is exactly what the AI gets to see each step: 31 numbers per car,
> such as how far the road's edges are, how fast the car is going, and how much the road bends
> over the next 150 m. Each number is scaled to about −1…1 so the neural network can learn from
> it. You can turn any input off in the settings. A fingerprint (the *digest*) of the whole
> list is kept with every trained AI, so it can never be fed numbers it wasn't trained on.

The observation is built by `ObservationBuilder` in `mlracecar.env.observations`, from a
`Snapshot` and the action each car was last given. The ideas behind it are in
[RL fundamentals](rl-guide.md), section 3.

```python
from mlracecar.env.observations import ObservationBuilder

builder = ObservationBuilder(track, config.vehicle.to_params(), config.observation,
                             config.sensors.to_settings())
observations = builder.build(world.snapshot, previous_actions)  # (cars, 31), float32
builder.spec.labels   # a name for every column: "rays[0]", ..., "curvature[7]"
builder.spec.digest   # the fingerprint, 64 hex digits
```

## The inputs

In this order; each can be turned off (see below). With the default settings there are 31 values.

| Input | Values | What it is | Scaled so that |
|-------|-------:|------------|----------------|
| `rays` | 15 | Distance to the road's edge along each ray, right to left ([Sensors](sensors.md)). | 0 = touching the edge, 1 = nothing within the 100 m range. |
| `speed` | 1 | Forward speed. | 100 m/s (360 km/h) = 1. The default car's top speed is 0.84. |
| `heading` | 2 | The angle between the car and the road, as its sine and cosine. | Along the road: (0, 1). Turned left across it: (1, 0). Backwards: (0, −1). |
| `offset` | 1 | Distance from the middle of the road, left positive. | ±1 at the road's edges. |
| `yaw_rate` | 1 | How fast the car is turning. | 2 radians a second = 1, about the most the default car's grip allows. |
| `steering` | 1 | Where the front wheels point now. | ±1 at full lock, left positive. |
| `previous_action` | 2 | The `[steer, pedal]` the car was given last step. | As given: −1…1. |
| `curvature` | 8 | How much the road bends over each of 8 equal stretches covering the next 150 m. | A bend of 10 m radius = 1, left positive. A 12 m hairpin is about 0.8, a 100 m sweeper 0.1. |

Values that could grow without limit (speed, offset, yaw rate, curvature) are clipped to ±2,
so even a car flung far onto the grass gives bounded numbers. A test throws random cars at it
(anywhere within a kilometre of the track, at up to 120 m/s, spinning at up to 5 radians a
second) and checks that every value is finite and within its bounds.

**Why sine and cosine for the heading?** As an angle, "almost backwards turning left" and
"almost backwards turning right" would be +3.1 and −3.1, far apart, although the car barely
moved. Sine and cosine change smoothly all the way round.

**Why each stretch's *average* bend, not the bend at a point?** A point sample could fall just
before or after a short, tight corner and miss it. The average over a stretch (how much the
road's direction changes along it, per metre) counts every bend in it.

## Settings

The `observation` section of the [settings](configuration.md). Each input is `true` or `false`:

```bash
uv run racecar config --set observation.curvature=false
```

| Setting | Default | Meaning |
|---------|---------|---------|
| `observation.rays` … `observation.curvature` | `true` | Whether each input is in. At least one must be. |
| `observation.lookahead` | 150 | How far ahead the bends are measured, in metres. |
| `observation.lookahead_points` | 8 | How many equal stretches that distance is split into. |

How many rays there are, how wide they fan, and how far they reach are the `sensors` settings.

150 m is enough for the default car to brake from its top speed (303 km/h) to hairpin speed,
which takes about 115 m, with room to turn in. The rays only reach 100 m.

## The spec and its digest

`builder.spec` is an `ObservationSpec`: the inputs in order, each with a label for every value,
its bounds (`spec.low`, `spec.high`, for the Gymnasium observation space), and the constants
that shape it (the ray settings, the scales, the stretch length).

`spec.digest` is a SHA-256 hash of all of that. It changes whenever anything that changes the
numbers changes: an input turned on or off, a different number of rays or range, a different
lookahead. It doesn't depend on the track. When a model is trained (M4), its digest is saved
with it, and loading it into an environment whose observations have a different digest fails
with a clear error instead of producing a confused driver.

A test pins the digest of the default settings. If it fails, the default observation changed,
and every model trained before can't be used any more: say so in the pull request. If an
input's meaning changes without any setting changing (a new scale, say), raise `SPEC_VERSION`
so the digest changes too.
