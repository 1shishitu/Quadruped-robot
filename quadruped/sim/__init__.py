"""Simulation environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quadruped.sim.mujoco_env import MuJoCoEnv

__all__ = ["MuJoCoEnv"]


def __getattr__(name: str):
    if name == "MuJoCoEnv":
        from quadruped.sim.mujoco_env import MuJoCoEnv

        return MuJoCoEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
