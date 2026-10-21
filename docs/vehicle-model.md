# Car physics

> **In plain words:** Every car is moved by the same small set of rules, 120 times per
> simulated second. The driver (you, or the AI) only sets the steering wheel and one pedal. The
> wheels turn towards where the driver steers, the engine or the brakes change the speed, and
> the car moves in the direction its wheels point. The one exception is when it's going too
> fast for the turn: then the tyres can't hold it and it runs wide.

This page describes the first car model, the **kinematic bicycle** (`KinematicBicycle` in
`mlracecar.core.vehicle.kinematic`). The choices behind it are in
[ADR-0014](adr/0014-kinematic-car-model.md), and its numbers are settings
(see [Settings](configuration.md)).

## What the default car can do

The default car is an arcade car: easy to drive with a keyboard, not realistic. It's 1300 kg
and 250 kW, its tyres grip at up to 3 g, its steering is quick, and its air drag is high, so it
tops out around 210 km/h. Lifting off the throttle slows it straight away (engine braking), so
you can lift for a corner instead of braking. You and the AI drive this same car
([ADR-0016](adr/0016-one-car-tuned-for-the-keyboard.md)).

It got here in steps, each from keyboard test drives. A plain sporty road car (200 kW, grip
1.0, 13 kN brakes) was slow to speed up and slow down, and couldn't turn hard enough to make
the corners. A light race car (grip 2, 37 kN brakes, 303 km/h) fixed that, but felt like a
racing simulator on a twisty track: it hardly slowed when you lifted off, so you braked all the
time. The arcade car is the tuning that drove well.

| | |
|---|---|
| 0–100 km/h | 3.3 s |
| 0–200 km/h | 18 s |
| Top speed | 209 km/h |
| Braking 100–0 km/h | 11.8 m |
| Braking 200–0 km/h | 46 m |
| Lifting off for 1 s, from 100 / 200 km/h | down to 83 / 176 km/h |
| Fastest through a bend of radius 12 m / 50 m / 200 m | 68 / 138 / 276 km/h |
| Tightest turn (rear-axle radius, at walking pace) | 4.7 m |
| Steering lock to lock | 0.17 s |

The tests keep the top speed at about 210 km/h and 0–100 km/h between 2.5 and 4 seconds, so a
change to the physics that makes the car feel different shows up straight away.

## The driver's controls

Each car gets an action of two numbers in `[-1, 1]` every step:

- **`steer`**: +1 is full lock to the left, -1 full lock to the right.
- **`pedal`**: +1 is full throttle, -1 full braking, 0 coasting.

Values outside `[-1, 1]` are clipped. Anything that isn't a finite number (NaN, infinity) is
refused with an error, so a broken AI can't silently wreck its car.

## The state of a car

All cars are stored together, one array per quantity
([ADR-0005](adr/0005-batched-multi-car-simulation.md)):

| Quantity | Meaning |
|----------|---------|
| `x`, `y` | The car's centre, in metres: the middle of the wheelbase. |
| `yaw` | Heading, radians counter-clockwise from +x. |
| `vx`, `vy` | Speed forward and to the left, in the car's own frame (m/s). |
| `yaw_rate` | How fast the heading turns (rad/s, positive = left). |
| `steer` | The front wheels' angle (radians, positive = left). |

## One step, in order

Each step of length `dt` (1/120 s by default) does this, for all cars at once:

1. **Steering.** The front wheels turn towards `steer · max_steer`, but by at most
   `steer_rate · dt`. Real wheels can't snap from lock to lock.
2. **Speed.** The forces along the car add up to an acceleration:

    - **Engine:** `pedal · min(max_drive_force, max_power / v)`, for `pedal > 0`. Below
      75 km/h the engine pushes with its full force; above that its power runs out, and the push
      falls as the speed rises.
    - **Brakes:** `pedal · max_brake_force`, for `pedal < 0`.
    - **Engine braking:** `engine_braking`, for `pedal ≤ 0`, fading to nothing at full
      throttle: `(1 − pedal) · engine_braking`, for `pedal > 0`. Lifting off the throttle
      slows the car. It fades instead of switching off at exactly 0, so an AI's pedal gets the
      same engine braking as a released key.
    - **Air drag:** `½ · ρ · Cd · A · v²`, with air density `ρ = 1.225 kg/m³`.
    - **Rolling resistance:** `rolling_resistance · m · g`.

    The new speed is `v + (engine or brakes − engine braking − drag − rolling) / m · dt`. It
    never goes below zero: brakes and resistance stop the car, but don't drive it backwards.
    There is no reverse gear.

3. **Grip.** The tyres can hold at most `grip · g` sideways. A car turning about a point
   `R = wheelbase / tan(angle)` from its rear axle needs up to `v² / R` sideways, so at speed
   `v` the wheel angle the car can follow is limited to

    `tan(angle) ≤ grip · g · wheelbase / v²`.

    Steering harder than that makes no difference: the car turns at the limit and runs wide
    (understeer). That's why the car has to brake for corners.

4. **Turning.** Each pair of wheels acts as one, like a bicycle, and the tyres don't slip, so the
   rear wheels roll straight ahead. The car turns about a point level with the rear axle, and its
   centre (halfway along the wheelbase) moves at a small angle to the heading, the **sideslip
   angle** `β = atan(tan(angle) / 2)`. The heading turns at

    `yaw_rate = v · cos(β) · tan(angle) / wheelbase`.

5. **Moving.** `yaw += yaw_rate · dt`, then the centre moves `v · dt` in the direction
   `yaw + β`.

Steps 2 and 5 use the *new* speed and heading. This is **semi-implicit Euler**, which stays
stable at the default step length for every speed the car can reach.

## What the model leaves out

- At the grip limit the car keeps all its speed, where real tyres would scrub some off.
- Braking or accelerating hard doesn't reduce how hard the car can corner (no "friction circle").
- No weight transfer, suspension, gears, or tyre wear.

The dynamic bicycle model (M2-3) adds tyre slip, so understeer, oversteer, and drifting come out
of the physics instead.
