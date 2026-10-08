"""Reinforcement-learning environments (Gymnasium / PettingZoo), observations, and rewards.

May import: config, io, core. Never imports render: training runs headless.

Importing this package registers ``MLRacecar-v0`` with Gymnasium:
``gymnasium.make("MLRacecar-v0", track="tracks/technical.json")``. Its entry point is in
`mlracecar.play.environment`, which can hand the environment a viewer for render modes.
"""

import gymnasium

ENV_ID = "MLRacecar-v0"
"""The name `RacingEnv` is registered under with Gymnasium."""

if ENV_ID not in gymnasium.registry:
    gymnasium.register(id=ENV_ID, entry_point="mlracecar.play.environment:make_racing_env")
