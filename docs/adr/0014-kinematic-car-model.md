# ADR-0014: A kinematic car with grip-limited turning and no reverse gear

- **Status:** Accepted
- **Date:** 2026-10-07
- **Related:** ticket M2-2 (#18), M2-3 (#19), [ADR-0005](0005-batched-multi-car-simulation.md), [Car physics](../vehicle-model.md)

> **In plain words:** The first car model is the simple kind where the car goes where its
> wheels point. On its own, that kind of car could take any corner at any speed, and an AI
> would learn to never brake. So the tyres get a grip limit: too fast for a corner, the car
> runs wide. The car's position is its middle, and it has no reverse gear.

## Context

The kinematic bicycle model (architecture section 4.3) is simple and stable, which makes it
a good first model for training. But it has no tyres: the car follows the circle its wheels
point at, at any speed. At 250 km/h it could turn as tightly as when parking, sideways forces
of tens of g included. An RL agent finds that immediately, and the racing line, braking points,
and corner speeds stop mattering.

The model also needs a reference point (where "the car's position" is) and a rule for
negative pedal at a standstill.

## Options considered

**The turning problem**

1. **Pure kinematic model.** Textbook, but corners don't need braking (see above).
2. **Scale the steering down with speed**, as many arcade games do. It works, but the
   scale curve is made up, and it says nothing about grip, which #19 (tyre slip) and M5
   (domain randomization) need.
3. **Limit the turn by grip:** the tyres hold at most `grip · g` sideways, so at speed `v`
   the car can't follow a tighter circle than `v² / (grip · g)`. Steering beyond that
   turns no tighter (understeer). This is one physical parameter, the friction coefficient,
   which the dynamic model will reuse.

**The reference point**

1. **Rear axle.** The kinematic equations are simplest there, but the renderer, race rules,
   sensors, and the dynamic model all want the car's middle.
2. **Centre of the wheelbase**, where the body is centred and the centre of mass is assumed to
   be. The centre then moves at a small sideslip angle to the heading.

**Negative pedal at a standstill**

1. **Reverse gear.** More states to learn and test; racing doesn't need it.
2. **Brakes hold the car.** A car stuck facing a wall is the race rules' job (#21: off-track
   reset).

## Decision

`KinematicBicycle` limits the turn by grip (option 3), uses the centre of the wheelbase as the
car's position, and has no reverse gear. It steps with semi-implicit Euler: new steering angle
and speed first, then the turn and the move. Actions are checked (finite, shape `(N, 2)`) and
clipped to `[-1, 1]`.

## Consequences

- **Positive:** cornering speed depends on the corner's radius and the tyres' grip, as in a
  real car, so braking for corners matters from the first training run. The new `grip`
  setting (1.0 for the default car then; see `configs/default.yaml` now) is the friction
  coefficient #19 needs. The rear axle
  still follows a circle of radius `wheelbase / tan(steer)` when grip allows (tested).
- **Negative / costs:** at the grip limit the car keeps all its speed, where real tyres would
  scrub some off. Braking and accelerating don't reduce cornering grip (no friction circle).
  The dynamic model (#19) covers both.
- **Follow-ups:** if keyboard driving (#23) needs a way out of dead ends, reverse can come
  back as "hold the brake at a standstill"; the world (#20) and the race rules (#21) use the
  centre as the car's position.
