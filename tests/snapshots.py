"""Test-only helpers for snapshots: comparing them, and recording a run driven alone."""

from dataclasses import fields

import numpy as np

from mlracecar.agents.base import Agent
from mlracecar.config.models import RacecarConfig
from mlracecar.core.snapshot import Snapshot
from mlracecar.core.track.model import Track
from mlracecar.env.racing import RacingEnv


def assert_same_snapshots(first: list[Snapshot], second: list[Snapshot]) -> None:
    """Snapshot by snapshot, field by field, bit for bit; events too (NaN equal to NaN)."""
    assert len(first) == len(second)
    for one, other in zip(first, second, strict=True):
        assert (one.tick, one.time) == (other.tick, other.time)
        assert repr(one.events) == repr(other.events)
        for part in ("cars", "race"):
            for field in fields(getattr(one, part)):
                np.testing.assert_array_equal(
                    getattr(getattr(one, part), field.name),
                    getattr(getattr(other, part), field.name),
                    err_msg=f"{part}.{field.name}",
                )


def live_snapshots(
    track: Track, config: RacecarConfig, seed: int, start: str, agent: Agent
) -> list[Snapshot]:
    """The world's snapshots of one run driven alone in a `RacingEnv`, from start to end."""
    env = RacingEnv(track, config)
    observation, _ = env.reset(seed=seed, options={"start": start})
    agent.reset()
    assert env.world is not None
    snapshots = [env.world.snapshot]
    while True:
        observation, _, terminated, truncated, _ = env.step(agent.act(observation[None])[0])
        snapshots.append(env.world.snapshot)
        if terminated or truncated:
            return snapshots
