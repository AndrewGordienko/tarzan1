"""Predicate verification + failure detection -- ON BELIEF, not ground truth.

The executor uses this to decide subgoal/goal completion and to detect
recovery-worthy events. Every check reads the agent's BeliefState (estimated
poses, inferred grasp), so a wrong or uncertain belief can (correctly) mislead
the agent -- which is the point. Ground-truth success and safety are scored
separately by the benchmark Scorer.
"""
from __future__ import annotations

from ..compiler.task_graph import Predicate
from ..geometry import apply, dist_xy, dist_xyz
from ..agent.belief import BeliefState

NEAR_XY = 0.06
AT_REL_TOL = 0.05


def predicate_holds(p: Predicate, b: BeliefState, rel_map: dict | None = None) -> bool:
    """`p.args` are track IDs (resolved from roles via correspondence)."""
    if p.name == "grasped":
        return b.grasped == p.args[0]
    o = p.args[0]
    if o not in b.objects:
        return False
    obj = b.objects[o]
    if p.name == "on_table":
        return abs(obj.pose[2] - (b.table_z + obj.size[2] / 2)) < 0.02 and b.grasped != o
    if p.name == "near":
        r = p.args[1]
        return r in b.objects and dist_xy(obj.pose, b.objects[r].pose) < NEAR_XY
    if p.name == "at_rel":
        r = p.args[1]
        if r not in b.objects or b.grasped == o or rel_map is None:
            return False
        rel = rel_map.get((o, r))
        if rel is None:
            return False
        return dist_xy(obj.pose, apply(b.objects[r].pose, rel)) < AT_REL_TOL
    if p.name == "on_top":
        r = p.args[1]
        if r not in b.objects:
            return False
        support = b.objects[r]
        return (dist_xy(obj.pose, support.pose) < NEAR_XY
                and obj.pose[2] > support.pose[2] + support.size[2] / 4
                and b.grasped != o)
    return False


class Verifier:
    def __init__(self, goal: frozenset, rel_map: dict | None = None):
        self.goal = goal
        self.rel_map = rel_map or {}

    def goal_satisfied(self, b: BeliefState) -> bool:
        return bool(self.goal) and all(predicate_holds(p, b, self.rel_map) for p in self.goal)

    def progress(self, b: BeliefState) -> float:
        if not self.goal:
            return 0.0
        return sum(predicate_holds(p, b, self.rel_map) for p in self.goal) / len(self.goal)

    def detect_failure(self, prev: BeliefState, cur: BeliefState, expected_grasp) -> str | None:
        """Recovery-worthy event, judged from belief. An intended release opens
        the gripper (closed->0); a knocked-out grasp leaves it commanded closed."""
        if (expected_grasp is not None and prev.grasped == expected_grasp
                and cur.grasped is None and cur.gripper_closed >= 0.6):
            return f"dropped:{expected_grasp}"
        # a goal object we believed placed is no longer where it should be
        return None
