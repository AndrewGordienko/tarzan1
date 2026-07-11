"""Backend-agnostic simulator interface.

Everything above this line in the stack (Stages A-E, metrics, the world model)
talks only to ``SimBackend`` + ``SimState`` + ``Observation``. The ToyTabletopSim
CPU backend and a future ManiSkillBackend both implement this, so switching from
laptop development to a GPU cloud fleet is a one-line backend swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np

from ..geometry import Pose, pose


@dataclass
class SimObject:
    name: str
    shape: str = "box"            # box | cylinder
    size: np.ndarray = field(default_factory=lambda: np.array([0.04, 0.04, 0.04]))
    mass: float = 0.1
    friction: float = 0.6
    color: str = "red"
    marker: str = "unknown"       # view-dependent visual label; not an object name
    pose: Pose = field(default_factory=pose)

    def copy(self) -> "SimObject":
        return replace(self, size=self.size.copy(), pose=self.pose.copy())


@dataclass
class SimState:
    objects: dict[str, SimObject]
    gripper: Pose                       # x, y, z, yaw of the TCP
    gripper_closed: float = 0.0         # 0 open .. 1 closed
    grasped: str | None = None          # name of held object, if any
    table_bounds: tuple = (-0.3, 0.3, -0.3, 0.3)   # xmin, xmax, ymin, ymax
    table_z: float = 0.0
    fallen: set = field(default_factory=set)       # objects knocked off the table
    t: int = 0

    def copy(self) -> "SimState":
        return SimState(
            objects={k: v.copy() for k, v in self.objects.items()},
            gripper=self.gripper.copy(), gripper_closed=self.gripper_closed,
            grasped=self.grasped, table_bounds=self.table_bounds,
            table_z=self.table_z, fallen=set(self.fallen), t=self.t)


@dataclass
class Observation:
    """What Stage-A perception is allowed to see.

    On the toy backend these are privileged (ground-truth object tracks) but
    corrupted by pose noise that scales with the randomized `lighting` level, so
    downstream code must already tolerate the imperfect tracks a real video
    encoder would produce.
    """
    object_tracks: dict[str, Pose]      # per-object observed pose
    gripper: Pose
    gripper_closed: float
    contact: dict[str, bool]            # observed gripper<->object contact
    t: int


@dataclass
class StepInfo:
    collision: bool = False
    force_violation: bool = False
    irreversible: bool = False          # an object left the table this step
    events: list = field(default_factory=list)


# High-level action passed to a backend: a target TCP pose + gripper command.
@dataclass
class Action:
    target: Pose
    gripper_close: float                # desired 0..1


class SimBackend(Protocol):
    def reset(self, state: SimState) -> Observation: ...
    def step(self, action: Action) -> tuple[Observation, StepInfo]: ...
    def state(self) -> SimState: ...
    def observe(self) -> Observation: ...
