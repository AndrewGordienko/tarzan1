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
    marker: str = "unknown"        # only retained after a view exposes it
    pos_std: float = 0.02          # position uncertainty (metres)
    size_std: float = 0.02         # size-estimate uncertainty (metres); falls ~1/sqrt(N)
                                   # as independent observations are fused
    size_var: np.ndarray | None = None  # per-axis attribute posterior covariance
    last_seen: int = 0             # frame index of last direct observation
    visible: bool = True           # observed this frame?
    # An association can be spatially plausible but still join a different
    # object's attributes to this trajectory.  This is deliberately retained in
    # the agent-visible belief so resolution can decline to commit on it.
    association_contested: bool = False
    association_stable_frames: int = 0

    def feature(self) -> np.ndarray:
        """Appearance/geometry signature used for demo<->eval correspondence.
        Deliberately name-free: size, shape one-hot, DETERMINISTIC color code
        (Python's builtin hash varies between processes and would break
        reproducibility)."""
        shape_id = 0.0 if self.shape == "box" else 1.0
        return np.array([self.size[0], self.size[2], shape_id, color_code(self.color),
                         marker_code(self.marker)])

    def feature_var(self) -> np.ndarray:
        """Per-feature uncertainty for likelihood-based correspondence.

        Shape/color are categorical rather than repeatedly averaged metric
        measurements.  The matcher currently conditions only on size axes, but
        keeps finite values here so extensions can opt in explicitly.
        """
        sv = self.size_var if self.size_var is not None else np.full(3, self.size_std ** 2)
        marker_var = 1e-6 if self.marker != "unknown" else 1.0
        return np.array([sv[0], sv[2], 1e-6, 1e-6, marker_var], dtype=float)

    def copy(self) -> "BeliefObject":
        return replace(self, pose=self.pose.copy(), size=self.size.copy(),
                       size_var=None if self.size_var is None else self.size_var.copy())


# Fixed palette -> stable [0,1) code, independent of process hash seed.
_PALETTE = ("red", "green", "blue", "yellow", "purple", "orange", "cyan",
            "magenta", "unknown")


def color_code(color: str) -> float:
    try:
        return _PALETTE.index(color) / len(_PALETTE)
    except ValueError:
        return (len(_PALETTE) - 1) / len(_PALETTE)


_MARKERS = ("alpha", "beta", "gamma", "unknown")


def marker_code(marker: str) -> float:
    try:
        return _MARKERS.index(marker) / len(_MARKERS)
    except ValueError:
        return (len(_MARKERS) - 1) / len(_MARKERS)


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
