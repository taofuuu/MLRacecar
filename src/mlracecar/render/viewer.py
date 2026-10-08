"""`RaceViewer`: shows an environment's race, in a window or as pictures.

`mlracecar.env.racing.RacingEnv` never draws (training runs without pygame); when it's made
with a render mode, it hands each snapshot to one of these. It draws with `RaceRenderer`,
following the car, with the distance rays on, so you see what the agent sees.
"""

import numpy as np
import pygame
from numpy.typing import NDArray

from mlracecar.core.geometry import FloatArray
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.core.vehicle.params import VehicleParams
from mlracecar.render.race import Overlay, RaceRenderer

SIZE = (960, 600)
"""The window's or picture's width and height, in pixels."""


class RaceViewer:
    """Draws one track's race.

    Args:
        track: The track.
        car: The car, for its size.
        mode: ``"human"`` for a window shown in real time, ``"rgb_array"`` for pictures.
        fps: Frames a second in the window: the environment's decisions per second.
    """

    def __init__(
        self,
        track: Track,
        car: VehicleParams,
        mode: str,
        fps: float,
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"mode must be 'human' or 'rgb_array', got {mode!r}")
        self.mode = mode
        self.fps = fps
        self.renderer = RaceRenderer(track, (car.length, car.width), SIZE)
        self.renderer.overlays = {Overlay.RAYS}
        self._window: pygame.Surface | None = None
        self._clock = pygame.time.Clock()

    def render(self, snapshot: Snapshot, rays: FloatArray | None) -> NDArray[np.uint8] | None:
        """A picture of the race, or ``None`` after showing it in the window (``human``)."""
        if self.mode == "rgb_array":
            return self.renderer.render(snapshot, rays)
        if self._window is None:
            pygame.display.init()
            self._window = pygame.display.set_mode(SIZE)
            pygame.display.set_caption("MLRacecar")
        pygame.event.pump()  # keeps the window responsive
        self.renderer.draw(self._window, snapshot, rays)
        pygame.display.flip()
        self._clock.tick(self.fps)
        return None

    def close(self) -> None:
        """Close the window, if one is open."""
        if self._window is not None:
            pygame.display.quit()
            self._window = None
