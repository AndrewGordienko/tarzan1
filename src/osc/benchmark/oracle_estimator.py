"""OracleEstimator: a privileged state estimator for the belief-vs-oracle ablation.

It ignores percepts and reports ground truth with ~zero uncertainty and the true
grasp. Used ONLY inside the ablation harness to measure the upper bound / the
cost of imperfect perception. It is never part of the deployed agent, and the
architectural no-privileged-access test excludes ablation code paths.
"""
from __future__ import annotations

from ..agent.belief import BeliefObject, BeliefState


class OracleEstimator:
    def __init__(self, backend):
        self._backend = backend
        self.grasped = None

    def update(self, percept) -> BeliefState:
        s = self._backend.state()               # privileged: ablation only
        objs = {}
        for name, o in s.objects.items():
            if name in s.fallen:
                continue
            objs[name] = BeliefObject(track_id=name, pose=o.pose.copy(),
                                      size=o.size.copy(), shape=o.shape,
                                      color=o.color, pos_std=0.001,
                                      last_seen=s.t, visible=True)
        self.grasped = s.grasped
        b = BeliefState(objects=objs, gripper=s.gripper.copy(),
                        gripper_closed=s.gripper_closed, grasped=s.grasped,
                        grasp_confidence=1.0, table_z=s.table_z, t=s.t)
        return b
