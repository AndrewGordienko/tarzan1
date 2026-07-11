"""Task scenes + the scripted oracle that records the single demonstration.

For the vertical slice we ship one task: STACK (place cube A on cube B). The
oracle is an independent scripted expert used ONCE, in the clean nominal scene,
to produce a demonstration. Stage A only ever sees the resulting object tracks
and contact signal -- never the oracle's code -- so compiling a task graph from
it and transferring to randomized scenes is a fair one-shot test.

Adding tasks 2..25 means adding scene dicts + an oracle each; the rest of the
pipeline (A-E, metrics) is untouched.
"""
from __future__ import annotations

import numpy as np

from .geometry import pose
from .perception.tracks import DemoTrace, extract_tracks
from .sim.base import Action
from .sim.randomize import nominal

STACK_SCENE = {
    "name": "stack",
    "table_bounds": (-0.3, 0.3, -0.3, 0.3),
    "objects": [
        {"name": "cube_a", "role": "manipuland", "base_pose": [-0.08, 0.05, 0.02, 0.0],
         "color": "red", "shape": "box"},
        {"name": "cube_b", "role": "target", "base_pose": [0.10, -0.03, 0.02, 0.0],
         "color": "blue", "shape": "box"},
    ],
    "roles": {"cube_a": "manipuland", "cube_b": "target"},
}


def _oracle_actions(state):
    """Scripted stack expert: approach A, grasp, lift, carry over B, place."""
    a = state.objects["cube_a"].pose
    b = state.objects["cube_b"].pose
    top_of_b = b[2] + state.objects["cube_b"].size[2] / 2 + state.objects["cube_a"].size[2] / 2
    waypoints = [
        (pose(a[0], a[1], a[2] + 0.09, 0.0), 0.0),   # hover over A
        (pose(a[0], a[1], a[2], 0.0), 0.0),          # descend
        (pose(a[0], a[1], a[2], 0.0), 1.0),          # close (grasp)
        (pose(a[0], a[1], a[2] + 0.10, 0.0), 1.0),   # lift
        (pose(b[0], b[1], a[2] + 0.12, 0.0), 1.0),   # carry over B (high)
        (pose(b[0], b[1], top_of_b, 0.0), 1.0),      # descend onto B
        (pose(b[0], b[1], top_of_b, 0.0), 0.0),      # release
        (pose(b[0], b[1], top_of_b + 0.09, 0.0), 0.0),  # retract
    ]
    return waypoints


def record_demo(scene: dict = STACK_SCENE, settle_steps: int = 6) -> DemoTrace:
    state, backend = nominal(scene)
    backend.reset(state)
    observations = [backend.observe()]
    for target, grip in _oracle_actions(state):
        for _ in range(settle_steps):
            obs, _ = backend.step(Action(target=target, gripper_close=grip))
            observations.append(obs)
    return extract_tracks(observations)
