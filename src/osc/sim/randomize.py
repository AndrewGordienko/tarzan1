"""Domain randomization with roles, distractors, and split control.

Ground-truth objects are named by role ("manip", "target", "distractor0", ...)
purely for the scorer; the agent never sees names (percepts are nameless). Role
objects keep a recognizable shape/size class in the base condition so
correspondence is solvable; the `new_instances` split perturbs their sizes to
test robustness. Distractors are added to force real manipuland/target selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import pose
from .base import SimObject, SimState
from .toy import ToyTabletopSim

COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
SHAPES = ["box", "cylinder"]


@dataclass
class RandomizationSpec:
    pose_jitter: float = 0.06
    role_size_jitter: float = 0.0          # >0 in the new-instances split
    mass_range: tuple = (0.05, 0.40)
    friction_range: tuple = (0.30, 0.90)
    actuator_delay_range: tuple = (0.10, 0.45)
    lighting_range: tuple = (0.20, 1.00)
    camera_jitter_range: tuple = (0.0, 0.01)
    n_distractors: int = 2
    randomize_distractor_shape: bool = True
    camera_model: bool = False
    # Hard-perception benchmark controls.  They change observable geometry, not
    # labels exposed to the agent.
    role_feature_mode: str = "native"  # native | height_only | identical
    distractor_min_separation: float = 0.09


def _free_pose(rng, bounds, taken, margin=0.09):
    xmin, xmax, ymin, ymax = bounds
    for _ in range(50):
        p = pose(rng.uniform(xmin + 0.05, xmax - 0.05),
                 rng.uniform(ymin + 0.05, ymax - 0.05), 0.02, rng.uniform(-np.pi, np.pi))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > margin for q in taken):
            return p
    return p


def randomize(scene: dict, spec: RandomizationSpec, seed: int):
    rng = np.random.default_rng(seed)
    bounds = tuple(scene.get("table_bounds", (-0.3, 0.3, -0.3, 0.3)))
    objects: dict[str, SimObject] = {}
    roles: dict[str, str] = {}
    taken = []

    for tmpl in scene["objects"]:
        base = np.asarray(tmpl["base_pose"], dtype=float).copy()
        base[:2] += rng.uniform(-spec.pose_jitter, spec.pose_jitter, size=2)
        base[3] = rng.uniform(-np.pi, np.pi)
        native = float(tmpl["size"]) * (1.0 + rng.uniform(-spec.role_size_jitter, spec.role_size_jitter))
        if spec.role_feature_mode == "height_only":
            size = np.array([0.043, 0.043, native])
        elif spec.role_feature_mode == "identical":
            size = np.array([0.043, 0.043, 0.043])
        else:
            size = np.array([native, native, native])
        objects[tmpl["name"]] = SimObject(
            name=tmpl["name"], shape=tmpl.get("shape", "box"),
            size=size,
            mass=float(rng.uniform(*spec.mass_range)),
            friction=float(rng.uniform(*spec.friction_range)),
            color=rng.choice(COLORS), pose=base)
        roles[tmpl["name"]] = tmpl["role"]
        taken.append(base)

    for k in range(spec.n_distractors):
        name = f"distractor{k}"
        ds = float(rng.uniform(0.03, 0.055))
        if spec.role_feature_mode == "identical":
            size = np.array([0.043, 0.043, 0.043])
            shape = "box"
        elif spec.role_feature_mode == "height_only":
            size = np.array([0.043, 0.043, ds])
            shape = rng.choice(SHAPES) if spec.randomize_distractor_shape else "box"
        else:
            size = np.array([ds, ds, ds])
            shape = rng.choice(SHAPES) if spec.randomize_distractor_shape else "box"
        objects[name] = SimObject(
            name=name, shape=shape, size=size,
            mass=float(rng.uniform(*spec.mass_range)),
            friction=float(rng.uniform(*spec.friction_range)),
            color=rng.choice(COLORS), pose=_free_pose(rng, bounds, taken,
                                                       margin=spec.distractor_min_separation))
        roles[name] = "distractor"
        taken.append(objects[name].pose)

    state = SimState(objects=objects, gripper=pose(0.0, -0.22, 0.16, 0.0),
                     table_bounds=bounds)
    backend = ToyTabletopSim(
        actuator_delay=float(rng.uniform(*spec.actuator_delay_range)),
        lighting=float(rng.uniform(*spec.lighting_range)),
        camera_jitter=float(rng.uniform(*spec.camera_jitter_range)),
        rng=rng, camera_model=spec.camera_model)
    return state, backend, roles


def nominal(scene: dict):
    """Clean env for recording the single demo: no distractors, no jitter."""
    spec = RandomizationSpec(pose_jitter=0.0, n_distractors=0,
                             actuator_delay_range=(0.2, 0.2),
                             lighting_range=(1.0, 1.0), camera_jitter_range=(0.0, 0.0),
                             mass_range=(0.1, 0.1), friction_range=(0.6, 0.6))
    return randomize(scene, spec, seed=0)
