# ADR-0016: One car for people and the AI, tuned to feel good on a keyboard

- **Status:** Accepted
- **Date:** 2026-10-21
- **Related:** ticket #113, #39 (first lap), [ADR-0014](0014-kinematic-car-model.md), [Car physics](../vehicle-model.md)

> **In plain words:** You and the AI drive exactly the same car, and that car is tuned to be
> easy and fun with a keyboard, like an arcade racer, rather than realistic. Nobody gets
> helpers the other doesn't have. Making the car easier means retraining the AI, which takes
> about a minute and a half.

## Context

Driving a hand-made track for #39 with the default car (the "light race car": 2 g of grip,
37 kN brakes, 303 km/h) felt like a racing simulator. Lifting off the throttle barely slowed
it (100 to 99 km/h in a second), so the driver braked all the time, and the twisty track came
at them too fast. #39 compares a person's best lap with the AI's, and the car has to be one a
person can drive well for that to mean anything.

## Options considered

1. **One car, made easier for everyone.** Change the default car; the AI is retrained on it.
   Lap times compare like with like. Costs a retrain (about 90 s on the CPU) and makes AIs
   trained before the change drive a different car.
2. **Driver aids for people only.** Keep the race car for the AI and add helpers to the
   keyboard (steering that eases off at speed, braking before corners). The AI is untouched,
   but a person's lap time is partly the helpers', and the AI keeps learning on a car people
   find hard.

## Decision

Option 1. People and the AI drive one car, and its defaults are tuned by keyboard test drives:
grip 3, steering twice as quick (360°/s), air drag 1.0 (top speed about 210 km/h), and a new
setting, engine braking (5000 N), that slows the car when the throttle is released. Driver aids
wait for M5, with the other human controls.

Engine braking fades out as the throttle is pressed instead of switching on only at exactly
zero throttle: a released key sends exactly 0, but the AI's pedal is a number that's almost
never exactly 0, so a switch would be a hidden aid for people.

The grip limit of [ADR-0014](0014-kinematic-car-model.md) stays: the car is easier, not
unlimited, so braking for corners still matters to the AI.

## Consequences

- **Positive:** a person's lap and the AI's lap are on the same car. The car is quicker to
  learn to drive, for people and probably for the AI.
- **Negative / costs:** AIs, runs, and golden test files from before this change are on the old
  car. Saved settings without `engine_braking` load it as 0 (not today's default), so testing
  or resuming an older AI still uses the car it learned on. The golden files were recorded again.
- **Follow-ups:** a chase camera for driving yourself (#114); driver aids and a gamepad (M5).
