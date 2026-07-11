"""PlanningModel: an approximate, skill-level forward model for imagined search.

This is intentionally a DIFFERENT model from the evaluation simulator. It does
not integrate per-step physics and never imports ToyTabletopSim. Instead it
predicts the *outcome* of each skill analytically (geometric/kinematic) and
scores risk from geometry + the estimated DynamicsContext:

  * collision_risk  : straight-line transport paths passing near other objects
                      at similar height (higher travel `lift` reduces it),
  * irreversible_risk: predicted object positions / paths near the (estimated)
                       table edge,
  * force_risk      : placing where an unstable stack is likely given estimated
                       friction,
  * uncertainty     : belief position std of involved objects + context spread.

Because the equations differ from the test environment's, a good plan here does
not trivially guarantee success there -- the search must be robust, not clairvoyant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose, apply, dist_xy, pose
from ..agent.belief import BeliefState
from ..agent.dynamics_context import DynamicsContext


@dataclass
class RolloutResult:
    success_prob: float
    collision_risk: float
    force_risk: float
    irreversible_risk: float
    uncertainty: float
    steps: int
    predicted: dict = field(default_factory=dict)   # track_id -> predicted pose


class PlanningModel:
    def __init__(self, context: DynamicsContext,
                 table_bounds_est=(-0.3, 0.3, -0.3, 0.3)):
        self.ctx = context
        self.bounds = table_bounds_est

    def rollout(self, belief: BeliefState, plan, goal_check) -> RolloutResult:
        # lightweight symbolic state: track -> pose, plus gripper/grasp
        state = {tid: o.pose.copy() for tid, o in belief.objects.items()}
        sizes = {tid: o.size.copy() for tid, o in belief.objects.items()}
        base_unc = {tid: o.pos_std for tid, o in belief.objects.items()}
        gripper = belief.gripper.copy()
        grasped = belief.grasped
        collision = force = irrev = 0.0
        steps = 0

        for si in plan:
            name = si.skill.name
            obj = si.params.get("object")
            steps += self._skill_steps(name)
            if name == "reach" and obj in state:
                gripper = pose(state[obj][0], state[obj][1], state[obj][2] + 0.09, 0)
            elif name == "grasp" and obj in state:
                if dist_xy(gripper, state[obj]) < 0.05:
                    grasped = obj
            elif name in ("move", "place"):
                ref = si.params.get("reference", "world")
                rel = si.params.get("rel", pose())
                ref_pose = pose() if ref == "world" else state.get(ref, pose())
                target = apply(ref_pose, rel)
                lift = si.params.get("lift", 0.06)
                # transport path risk: sweep from gripper xy to target xy
                collision += self._path_collision(gripper, target, lift, state, sizes, grasped, obj)
                irrev += self._edge_risk(target) + self._path_edge_risk(gripper, target)
                if grasped == obj and obj in state:
                    state[obj] = target.copy()
                    base_unc[obj] = base_unc.get(obj, 0.02) + 0.005
                gripper = pose(target[0], target[1], target[2], target[3])
                if name == "place":
                    grasped = None
                    force += self._stack_instability(target, ref, ref_pose, sizes, obj)

        # build a belief-like predicted state for the goal check
        pred = BeliefState(
            objects={tid: belief.objects[tid].copy() for tid in belief.objects},
            gripper=gripper, gripper_closed=0.0, grasped=grasped,
            table_z=belief.table_z, t=belief.t)
        for tid, p in state.items():
            if tid in pred.objects:
                pred.objects[tid].pose = p
        success = float(goal_check(pred))

        uncertainty = float(np.mean(list(base_unc.values()))) if base_unc else 0.02
        uncertainty += 0.5 * (self.ctx.delay_std + self.ctx.friction_std) / 2
        # tight-tolerance tasks lose success probability under high uncertainty
        success *= float(np.clip(1.0 - 1.5 * uncertainty, 0.2, 1.0))
        success *= float(np.clip(1.0 - 0.3 * irrev, 0.0, 1.0))
        return RolloutResult(success, collision, force, irrev, uncertainty, steps, state)

    # -- risk terms -------------------------------------------------------
    def _skill_steps(self, name: str) -> int:
        base = {"reach": 6, "grasp": 3, "move": 8, "place": 10, "release": 3}.get(name, 5)
        return int(base / max(0.3, 1.0 - self.ctx.actuator_delay))

    def _path_collision(self, start, target, lift, state, sizes, grasped, obj):
        risk = 0.0
        for tid, p in state.items():
            if tid in (grasped, obj):
                continue
            # transport happens at height ~ target.z + lift; only objects near
            # that height threaten the swept path
            travel_z = max(target[2] + lift, start[2])
            if abs(travel_z - p[2]) > (sizes[tid][2] / 2 + 0.03):
                continue
            d = _seg_point_dist(start[:2], target[:2], p[:2])
            clear = sizes[tid][0] / 2 + 0.03
            if d < clear:
                risk += (clear - d) / clear
        return risk

    def _edge_risk(self, p) -> float:
        xmin, xmax, ymin, ymax = self.bounds
        m = min(p[0] - xmin, xmax - p[0], p[1] - ymin, ymax - p[1])
        return float(np.clip((0.05 - m) / 0.05, 0.0, 1.0))

    def _path_edge_risk(self, start, target) -> float:
        return max(self._edge_risk(start), self._edge_risk(target))

    def _stack_instability(self, target, ref, ref_pose, sizes, obj) -> float:
        if ref == "world":
            return 0.0
        # overhang / low-friction stack risk
        off = dist_xy(target, ref_pose)
        base = sizes.get(ref, np.array([0.05, 0.05, 0.05]))[0] / 2
        overhang = float(np.clip((off - base * 0.4) / base, 0.0, 1.0))
        return overhang * float(np.clip(1.2 - self.ctx.friction_scale, 0.0, 1.0))


def _seg_point_dist(a, b, p) -> float:
    a, b, p = np.asarray(a), np.asarray(b), np.asarray(p)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))
