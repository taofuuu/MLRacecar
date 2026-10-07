# Sensors

> **In plain words:** The AI can't see the screen. Instead, each car has distance sensors, like
> the laser rangefinders on a self-driving car: invisible lines (rays) that fan out from the car
> and stop at the edge of the road. Each ray tells the AI how far the edge is in that direction.
> A short ray straight ahead means a bend is coming; a short ray to one side means the car is
> close to that edge. In `racecar drive`, press **4** to see them.

The sensors live in `mlracecar.core.sensors`. The RL environment (M3) reads them after every
step and gives the readings to the AI as part of what it sees.

## What the rays measure

- Every car gets the same fan of rays. They start at the car's centre and are spread evenly
  across the field of view, centred on the way the car points. The first ray is the rightmost
  (`RaySensor.angles`, measured counter-clockwise from the car's heading).
- A ray stops where it first meets either edge of the road. Its reading is the distance to that
  point, in metres. A ray that meets no edge within the range reads the range.
- The readings also come **normalized**: divided by the range, so they are always between 0
  (the edge is right there) and 1 (nothing in range). Neural networks learn best from inputs of
  about that size.
- Each ray's end point comes too, for drawing.

```python
import math

from mlracecar.core.sensors import RaySensor, RaySettings

sensor = RaySensor(track, RaySettings(count=15, field_of_view=math.pi, max_range=100.0))
readings = sensor.sense(world.snapshot)
readings.distance    # (cars, rays), metres
readings.normalized  # (cars, rays), 0..1
readings.end         # (cars, rays, 2), where each ray stops
```

## Settings

The `sensors` section of the [settings](configuration.md):

| Setting | Default | Meaning |
|---------|---------|---------|
| `sensors.rays` | 15 | How many rays fan out from the car. |
| `sensors.field_of_view` | 180 | The angle they fan across, in degrees: 180 is from straight left to straight right, 360 all the way round. |
| `sensors.range` | 100 | How far they reach, in metres. |

At racing speed (70 m/s, about 250 km/h) 100 m is about 1.4 seconds ahead. That's enough to
line up a corner. To brake in time from top speed the AI needs to know about bends further
ahead; the observation will add the road's curvature ahead for that (M3-2). More rays and a
longer range cost more time (see below).

## How it stays fast

The road's edges are lines with a point every half metre: about 14,000 pieces of edge on the GP
circuit. Testing every ray of every car against every piece would be far too slow, so the work
narrows down in three steps:

1. **Only the edges within reach.** A ray from a car on the road meets one of its own road's
   edges before it gets more than its range along the lap. So each car's rays are only tested
   against the edges from about `range + 10 m` behind its place on the lap to as far ahead
   (the *broad phase*).
2. **Only the blocks a ray passes close to.** The edges are cut into blocks of 16 pieces (8 m),
   each inside a circle worked out once per track. A ray is only tested against a block's pieces
   if it passes through that circle.
3. **Only the pieces a ray crosses.** A ray can only meet a piece of edge whose two ends lie on
   opposite sides of the ray's line. A quick sign test per point finds those pieces; only they
   get the exact calculation of where they meet.

Steps 1 and 2 only skip work that can't change the answer. A test compares the readings with
testing every piece of edge on the whole lap, for cars anywhere on the road and facing any way,
and they agree to within a billionth of a metre.

On the dev machine, reading 15 rays for each of 64 cars takes about 1.4 ms; the README has the
[full table](https://github.com/taofuuu/MLRacecar#speed).

## Limits

- **Cars on the grass see less.** Because of step 1, a car that has left the road doesn't see
  other parts of the track that pass close by, such as the far side of a hairpin. Rays from a
  car on the road can't reach those anyway: they meet the car's own road edge first.
- **Other cars are invisible to the rays.** Cars are ghosts that drive through each other until
  collisions arrive in M7.
