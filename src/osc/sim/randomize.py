"""Domain randomization: the knobs the benchmark requires per episode.

Given a nominal task scene and a seed, produce a randomized SimState + backend so
that the *same* compiled task program is forced to transfer across:
  * object instances (size / shape / color / mass),
  * initial poses and layout,
  * lighting and camera placement (perception noise),
  * friction, mass and actuator delay (dynamics).
Every draw is seed-deterministic so results are reproducible over fixed seeds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import pose
from .base import SimObject, SimState
from .toy import ToyTabletopSim

COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
SHAPES = ["box", "cylinder"]


@dataclass
class RandomizationSpec:
    pose_jitter: float = 0.08          # metres of layout jitter
    size_range: tuple = (0.03, 0.055)
    mass_range: tuple = (0.05, 0.4)
    friction_range: tuple = (0.3, 0.9)
    actuator_delay_range: tuple = (0.1, 0.45)
    lighting_range: tuple = (0.2, 1.0)
    camera_jitter_range: tuple = (0.0, 0.01)
    randomize_instances: bool = True


def randomize(scene: dict, spec: RandomizationSpec, seed: int) -> tuple[SimState, ToyTabletopSim]:
    """`scene` is the nominal task description: table bounds + a list of object
    templates {name, role, base_pose, color, shape}. Returns a randomized state
    and a backend whose physics params are drawn from `spec`."""
    rng = np.random.default_rng(seed)
    objects: dict[str, SimObject] = {}
    for tmpl in scene["objects"]:
        base = np.asarray(tmpl["base_pose"], dtype=float)
        jitter = np.zeros(4)
        jitter[:2] = rng.uniform(-spec.pose_jitter, spec.pose_jitter, size=2)
        jitter[3] = rng.uniform(-np.pi, np.pi) if spec.randomize_instances else 0.0
        s = rng.uniform(*spec.size_range)
        color = rng.choice(COLORS) if spec.randomize_instances else tmpl.get("color", "red")
        shape = rng.choice(SHAPES) if spec.randomize_instances else tmpl.get("shape", "box")
        objects[tmpl["name"]] = SimObject(
            name=tmpl["name"], shape=shape,
            size=np.array([s, s, s]),
            mass=float(rng.uniform(*spec.mass_range)),
            friction=float(rng.uniform(*spec.friction_range)),
            color=color, pose=base + jitter)

    state = SimState(objects=objects,
                     gripper=pose(0.0, -0.2, 0.15, 0.0),
                     table_bounds=tuple(scene.get("table_bounds", (-0.3, 0.3, -0.3, 0.3))))
    backend = ToyTabletopSim(
        actuator_delay=float(rng.uniform(*spec.actuator_delay_range)),
        lighting=float(rng.uniform(*spec.lighting_range)),
        camera_jitter=float(rng.uniform(*spec.camera_jitter_range)),
        rng=rng)
    return state, backend


def nominal(scene: dict) -> tuple[SimState, ToyTabletopSim]:
    """The clean, un-randomized environment used to record the single demo."""
    spec = RandomizationSpec(pose_jitter=0.0, randomize_instances=False,
                             actuator_delay_range=(0.2, 0.2),
                             lighting_range=(1.0, 1.0),
                             camera_jitter_range=(0.0, 0.0),
                             mass_range=(0.1, 0.1), friction_range=(0.6, 0.6))
    return randomize(scene, spec, seed=0)
