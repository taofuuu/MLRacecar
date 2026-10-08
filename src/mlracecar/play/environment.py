"""Makes `RacingEnv`s that can be watched: the target of ``gymnasium.make("MLRacecar-v0")``.

The environment never draws, so that training runs without pygame. When a render mode is asked
for, this hands it a `RaceViewer` to draw with, importing pygame only then.
"""

from mlracecar.config.models import RacecarConfig
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.env.racing import RacingEnv, TrackSource, Viewer


def make_racing_env(
    track: TrackSource, config: RacecarConfig | None = None, render_mode: str | None = None
) -> RacingEnv:
    """A `RacingEnv` on ``track``, drawn with a `RaceViewer` if there's a render mode."""
    return RacingEnv(track, config, render_mode=render_mode, viewer=_viewer)


def _viewer(track: Track, car: VehicleParams, mode: str, fps: float) -> Viewer:
    from mlracecar.render.viewer import RaceViewer  # pygame, only for someone watching

    return RaceViewer(track, car, mode, fps)
