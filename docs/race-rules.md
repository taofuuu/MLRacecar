# Race rules

> **In plain words:** The race rules keep track of each car: how far round the lap it is, how
> far it is from the middle of the road, and which way it points compared with the road. They
> also count laps and time them. A lap only counts if the car crossed every checkpoint (a line
> across the road) in order, so tricks like reversing back and forth over the finish line, or
> cutting across the grass, never count. They notice when a car leaves the road, and decide what
> happens then, and when it drives the wrong way.

The rules live in `mlracecar.core.race` and run after every driver decision (`World.step`).
Each `Snapshot` carries every car's race (`snapshot.race`) and what happened during that step
(`snapshot.events`).

## Where each car is

Each car is found on the road's centerline, which gives:

| Quantity | Meaning |
|----------|---------|
| `arc_length` | Metres along the lap from the start/finish line. |
| `distance` | Metres driven along the track since the car (re)started. Driving backwards subtracts. |
| `offset` | Metres from the middle of the road, positive to the left. |
| `heading_error` | The car's heading minus the road's: 0 is straight along the road, ±π backwards. |

Searching the whole lap for every car would be slow, so each car is only searched for near
where it was last time (amortized O(1) per car: the same small amount of work each step, however
long the track is). That also keeps a car on its own stretch of road: cutting across the grass
to a part of the track that passes close by doesn't move it along the lap.

## Checkpoints and laps

Checkpoints are lines across the road, about every 20 m (`Track.checkpoints`). Checkpoint 0 is
the start/finish line. The lines end at the road's edges.

- **A lap starts** when a car crosses the start/finish line forwards. A car on the grid, or one
  that starts anywhere else on the lap, begins its first lap when it reaches the line.
- **A lap ends** the next time the car crosses the line forwards. It is reported as a
  `LapCompleted` event either way, and it **counts** (it's *valid*) only if the car crossed every
  checkpoint in between, in order.
- **Crossing a checkpoint backwards undoes crossing it.** Reversing back and forth over the
  finish line therefore never counts a lap, and backing up over a checkpoint just means crossing
  it again.
- **Missing a checkpoint makes the lap invalid.** Since the lines end at the road's edges,
  cutting a corner past a checkpoint, or driving round one off the road, misses it.

Each car's race also records its valid laps (`laps`), its latest and best valid lap times
(`last_lap`, `best_lap`), and when its current lap started (`lap_start`).

## Times and sectors

A lap is split into three **sectors** that start at evenly spread checkpoints, as in real
racing. `LapCompleted` reports the lap time and each sector's time; they add up to the lap time.

The rules run once per driver decision (every 0.05 s by default), but times are worked out from
where exactly between two decisions the car crossed the line, so they are far more precise than
0.05 s.

## Off the road

A car is **off track** while its centre is off the road: further from the middle than half the
road's width there. With its centre off, half the car is on the grass. Each time a car leaves
the road, an `OffTrack` event says where, and `race.off_track` stays set until it's back.

What happens then is a setting, `race.off_track` (see [Settings](configuration.md)):

| `off_track` | What happens to a car off the road |
|-------------|-------------------------------------|
| `none` | Nothing. The event still says it happened. |
| `slowdown` (default) | The grass slows it down: it loses `race.grass_slowdown` m/s (6 by default) every second it's off the road, and never goes backwards. |
| `reset` | It's put back in the middle of the road where it left it, at rest, facing the right way. Its lap carries on. |
| `terminate` | Its run is over: it stops where it is and stays there, whatever the driver does, until it's reset. `race.out` says so, so that the RL environment can end the car's episode. |

## The wrong way

A car drives the **wrong way** while it moves backwards along the track faster than 1 m/s. A car
that just faces the wrong way, standing still or creeping, doesn't count. Each time it starts, a
`WrongWay` event says where, and `race.wrong_way` stays set until the car stops going backwards.
