"""Vehicle parameters: the fixed numbers that describe one kind of car.

Everything is in SI units: metres, kilograms, seconds, newtons, watts, and radians. The values
come from the settings files through `mlracecar.config`, which checks them, so the dynamics
models use them as they are.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParams:
    """One kind of car, in SI units. `mlracecar.config.models.VehicleConfig` builds one."""

    length: float
    """Length of the body in metres."""
    width: float
    """Width of the body in metres."""
    wheelbase: float
    """Distance from the front axle to the rear axle in metres."""
    mass: float
    """The car's mass, in kilograms."""
    max_steer: float
    """How far the front wheels turn at full lock, in radians."""
    steer_rate: float
    """How fast the front wheels can turn, in radians per second."""
    grip: float
    """Tyre friction coefficient: the tyres hold the car in a turn up to ``grip · g`` sideways."""
    max_drive_force: float
    """The most the engine can push, in newtons. At high speed `max_power` limits it instead."""
    max_power: float
    """Engine power in watts: at speed ``v`` the push is at most ``max_power / v``."""
    max_brake_force: float
    """The brakes' strongest push, in newtons."""
    drag_coefficient: float
    """Air resistance of the car's shape (Cd, no unit): drag = ½ · air density · Cd · area · v²."""
    frontal_area: float
    """Size of the car seen from the front, in square metres."""
    rolling_resistance: float
    """Rolling-resistance coefficient (no unit): the tyres resist with this share of the weight."""
