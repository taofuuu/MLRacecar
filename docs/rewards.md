# Rewards and episodes

> **In plain words:** While it trains, the AI gets points every step: 1 point for every 10 m
> it gets further round the lap, and 10 points off each time it leaves the road. Its run also
> ends there, and it starts again. A run also stops after 60 seconds, or if the car hasn't
> moved for 5 seconds. Other scoring parts (a cost for time, for driving the wrong way, for
> jerky controls, and a bonus per lap) are built but switched off; we turn one on only if
> watching the AI drive shows we need it.

The reward lives in `mlracecar.env.rewards`, the end-of-run rules in `mlracecar.env.episodes`.
The ideas behind both are in [RL fundamentals](rl-guide.md), sections 5 and 6.

## The reward

Every step, each car's reward is the sum of these terms. Each has a weight in the `reward`
settings; a weight of 0 switches the term off. Weights are never negative: each term already
knows whether it gives or takes points.

| Term | Points | Default weight | So by default |
|------|--------|---------------:|---------------|
| `progress` | + weight × metres gained along the lap (minus when going backwards) | 0.1 | 10 m = 1 point; a lap of the technical track ≈ 110 |
| `off_track` | − weight × times the car's centre left the road this step | 10 | −10, worth 100 m of progress |
| `time` | − weight × seconds (0.05 per step) | 0 | off |
| `wrong_way` | − weight × seconds spent driving the wrong way | 0 | off |
| `smoothness` | − weight × (change in steering² + change in pedal²) since the last step | 0 | off |
| `lap` | + weight × valid laps completed this step | 0 | off |

`RewardFunction(config, decision_dt)(before, after, actions, previous_actions)` returns
`Rewards(total, terms)`: the reward per car, and every term's points per car, so that it's
always clear which term drove what. `RewardTally` adds the terms up over each car's run; the
environment (M3-4) reports those sums in `info` when a run ends.

### Why progress, not speed

Rewarding speed would reward driving fast *anywhere*: in circles on the spot, zigzagging,
across the grass, even the wrong way round. Progress (`race.distance`) only counts metres
gained along the lap, so the only way to score is to get round the track, and going backwards
gives the points back. Speed still pays: the same metres sooner are worth more, because later
rewards are discounted, and more metres fit in a run's 60 seconds.

### Why only two terms to start with

Every extra term is another thing the AI can find a loophole in, and another reason it might
not learn. With two terms it's easy to tell what went wrong. If the trained car drives in a way
we don't like, the switched-off terms are the first tools to reach for: `smoothness` for
twitchy steering, `wrong_way` if it learns to reverse, `lap` to stress finishing laps.

## When a run ends

| Reason | When | Kind | Setting |
|--------|------|------|---------|
| Off the road | The car's centre leaves the road. | Terminated | `episode.end_off_track` (on) |
| Out | The race rules took the car out (`race.off_track: terminate`). | Terminated | always |
| Time limit | 60 seconds (1,200 steps) since the run began. | Truncated | `episode.time_limit` |
| Stuck | Slower than 1 m/s for 5 seconds in a row. | Truncated | `episode.stuck_time` |

*Terminated* means the run is really over: nothing can follow, so the future is worth 0.
*Truncated* means we stopped it although the car could have carried on, so learning still
counts on what would have followed. A run that ends both ways in the same step counts as
terminated. `EpisodeRules.check(after)` returns `Endings(terminated, truncated, reasons)` per
car; `start(cars)` restarts the clocks of cars whose new run begins.

Being stuck stops the run because early on, a random driver often just sits there; there's
nothing to learn from 60 seconds of that. A car starting from rest has 5 seconds to get going.

### Why leaving the road ends the run

With the gentle grass rule used for driving yourself (`race.off_track: slowdown`), going off
the road can pay. A scripted driver that takes corners too fast runs wide over the grass, yet
finishes laps sooner than a careful one. Measured on the technical track:

| Off-road rule | Careful driver | Reckless driver |
|---------------|----------------|-----------------|
| `slowdown`, grass 6 m/s² (the default for driving yourself) | 46.5 s laps | 43.6 s laps, not counted |
| `reset`: put back on the road, at rest | 46.5 s | 44.3 s |
| `slowdown`, grass 20 m/s² | 46.5 s | 44.3 s |
| `slowdown`, grass 40 m/s² | 46.5 s | 53.5 s |
| Run ends (`episode.end_off_track`) | 46.5 s | out after 24 s, with 480 m |

The AI would learn the same trick. Ending the run makes leaving the road always the worst
choice: it loses everything the car would have scored afterwards, plus the 10-point penalty.
It's an `episode` setting, not a race rule, so it applies whatever `race.off_track` says, and
`racecar drive` keeps the gentle rule.

## Settings

```yaml
reward:
  progress: 0.1  # Points per metre gained along the lap; going backwards loses them.
  off_track: 10  # Points lost each time the car's centre leaves the road.
  time: 0        # Points lost per second of racing.
  wrong_way: 0   # Points lost per second spent driving the wrong way.
  smoothness: 0  # Points lost per squared change of steering and pedal between decisions.
  lap: 0         # Points for each valid lap.

episode:
  end_off_track: true  # End the run as soon as the car's centre leaves the road.
  time_limit: 60       # The longest a run lasts, in seconds of racing.
  stuck_time: 5        # End the run after this many seconds in a row slower than 1 m/s.
```

For example, `--set reward.smoothness=0.05` switches on a small cost for jerky controls.
