"""BeliefState: the agent's estimate of the world.

Crucially this is NOT SimState. Objects are keyed by *anonymous, per-episode
track IDs* ("t0", "t1", ...) assigned by the estimator -- never by the
simulator's semantic names (cube_a/cube_b). Each estimate carries an uncertainty
(position std, in metres) that grows under occlusion / dropped frames and shrinks
when an object is observed. Grasp is *inferred*, not read from the sim, and comes
with a confidence.

Skills, the task planner, the verifier, the world model and the router all
consume BeliefState. Ground truth is available only to the benchmark scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..geometry import Pose, pose


@dataclass
class BeliefObject:
    track_id: str
    pose: Pose                      # estimated pose
    size: np.ndarray               # estimated bounding size
    shape: str = "box"             # estimated category
    color: str = "unknown"         # estimated appearance
    pos_std: float = 0.02          # position uncertainty (metres)
    last_seen: int = 0             # frame index of last direct observation
    visible: bool = True           # observed this frame?

    def feature(self) -> np.ndarray:
        """Appearance/geometry signature used for demo<->eval correspondence.
        Deliberately name-free: size, shape one-hot, coarse color hash."""
        shape_id = 0.0 if self.shape == "box" else 1.0
        col = float(abs(hash(self.color)) % 997) / 997.0
        return np.array([self.size[0], self.size[2], shape_id, col])

    def copy(self) -> "BeliefObject":
        return replace(self, pose=self.pose.copy(), size=self.size.copy())


@dataclass
class BeliefState:
    objects: dict[str, BeliefObject] = field(default_factory=dict)
    gripper: Pose = field(default_factory=pose)     # from proprioception (near-exact)
    gripper_closed: float = 0.0
    grasped: str | None = None                       # inferred held track id
    grasp_confidence: float = 0.0
    table_z: float = 0.0
    t: int = 0

    def copy(self) -> "BeliefState":
        return BeliefState(
            objects={k: v.copy() for k, v in self.objects.items()},
            gripper=self.gripper.copy(), gripper_closed=self.gripper_closed,
            grasped=self.grasped, grasp_confidence=self.grasp_confidence,
            table_z=self.table_z, t=self.t)

    def pose_of(self, track_id: str) -> Pose:
        return self.objects[track_id].pose

    def mean_uncertainty(self) -> float:
        if not self.objects:
            return 0.0
        return float(np.mean([o.pos_std for o in self.objects.values()]))
