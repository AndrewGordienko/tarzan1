"""Predicate verification + failure detection.

The verifier answers two questions the executor needs at every event boundary:
  1. Is the goal (or a subgoal's postcondition) satisfied in the current state?
  2. Has something gone wrong that warrants waking the planner (a lost grasp, an
     object knocked off-support, an irreversible loss)?
Only a "yes" to (2) triggers a replan -- this is the event-driven part.
"""
from __future__ import annotations

from ..compiler.task_graph import Predicate
from ..geometry import dist_xy
from ..sim.base import SimState

NEAR_XY = 0.06


def predicate_holds(p: Predicate, s: SimState) -> bool:
    if p.name == "grasped":
        return s.grasped == p.args[0]
    o = p.args[0]
    if o in s.fallen:
        return False
    obj = s.objects[o]
    if p.name == "on_table":
        return abs(obj.pose[2] - (s.table_z + obj.size[2] / 2)) < 0.01 and s.grasped != o
    if p.name == "near":
        return dist_xy(obj.pose, s.objects[p.args[1]].pose) < NEAR_XY
    if p.name == "on_top":
        support = s.objects[p.args[1]]
        return (dist_xy(obj.pose, support.pose) < NEAR_XY
                and obj.pose[2] > support.pose[2] + support.size[2] / 4
                and s.grasped != o)
    return False


class Verifier:
    def __init__(self, goal: frozenset):
        self.goal = goal

    def goal_satisfied(self, s: SimState) -> bool:
        return all(predicate_holds(p, s) for p in self.goal)

    def progress(self, s: SimState) -> float:
        if not self.goal:
            return 1.0
        return sum(predicate_holds(p, s) for p in self.goal) / len(self.goal)

    def detect_failure(self, prev: SimState, cur: SimState, expected_grasp) -> str | None:
        """Return a short event tag if a recovery-worthy event happened."""
        # irreversible loss of a goal object
        for p in self.goal:
            for a in p.args:
                if a in cur.objects and a in cur.fallen and a not in prev.fallen:
                    return f"lost:{a}"
        # unexpected grasp loss (a drop / slip). An *intended* release opens the
        # gripper (gripper_closed -> 0); a knocked-out grasp leaves it commanded
        # closed. Only the latter is a failure to recover from.
        if (expected_grasp is not None and prev.grasped == expected_grasp
                and cur.grasped is None and cur.gripper_closed >= 0.6):
            return f"dropped:{expected_grasp}"
        return None
