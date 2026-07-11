"""Narrow embodiment boundary for the v0.6 MuJoCo packing milestone."""

from .commands import SkillCommand
from .mujoco_adapter import ObservationFrame, TinyVLAMuJoCoAdapter

__all__ = ["SkillCommand", "ObservationFrame", "TinyVLAMuJoCoAdapter"]
