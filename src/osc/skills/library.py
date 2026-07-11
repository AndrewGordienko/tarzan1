"""Reusable motor-skill experts.

Each Skill is a small, independently testable, retargetable behaviour -- the
"skill expert" of the modular architecture. Here they are scripted closed-form
controllers (a param -> action mapping); in the full system each would be a small
learned policy sharing the common encoder/body model. What matters for the
architecture is that they are *reusable and composed by a router*, not one net
per task. A given task activates only a handful of them.

A Skill exposes:
  * `signature`: the predicate effect it can achieve (used by the router),
  * `precondition(state)`: cheap feasibility check,
  * `act(state, params)`: next motion target (Action) + a `done` flag,
so the same skill retargets to any object/pose by changing `params`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..geometry import Pose, apply, dist_xy, dist_xyz, pose
from ..sim.base import Action
from ..agent.belief import BeliefState

APPROACH_Z = 0.09       # hover height above a target before descending
STEP = 0.05             # nominal target step magnitude


@dataclass
class SkillInstance:
    skill: "Skill"
    params: dict
    label: str

    def act(self, state: BeliefState):
        return self.skill.act(state, self.params)

    def done(self, state: BeliefState) -> bool:
        return self.skill.done(state, self.params)


@dataclass
class Skill:
    name: str
    effect: str                         # predicate name this skill establishes
    _act: Callable
    _done: Callable
    _pre: Callable

    def act(self, state, params):
        return self._act(state, params)

    def done(self, state, params) -> bool:
        return self._done(state, params)

    def precondition(self, state, params) -> bool:
        return self._pre(state, params)


# -- primitive controllers -----------------------------------------------

def _reach_act(s: BeliefState, p):
    obj = s.objects[p["object"]]
    hover = pose(obj.pose[0], obj.pose[1], obj.pose[2] + APPROACH_Z, obj.pose[3])
    return Action(target=hover, gripper_close=0.0)

def _reach_done(s, p):
    obj = s.objects[p["object"]]
    hover = np.array([obj.pose[0], obj.pose[1], obj.pose[2] + APPROACH_Z])
    return float(np.linalg.norm(s.gripper[:3] - hover)) < 0.02


def _grasp_act(s: BeliefState, p):
    obj = s.objects[p["object"]]
    at = pose(obj.pose[0], obj.pose[1], obj.pose[2], obj.pose[3])
    close = 1.0 if dist_xy(s.gripper, obj.pose) < 0.02 and abs(s.gripper[2] - obj.pose[2]) < 0.04 else 0.0
    return Action(target=at, gripper_close=close)

def _grasp_done(s, p):
    return s.grasped == p["object"]


def _place_target(s: BeliefState, p) -> Pose:
    """Compute the placement target under the selected retargeting mode:
      * semantic  : solve the demonstrated RELATION on the CURRENT geometry
                    (target_z from current support/subject sizes), not a copied
                    metric offset -- this is the one-shot-learning thesis;
      * relative  : replay the exact demonstrated relative transform (v0.2);
      * absolute  : replay the demonstrated world pose (ignores the reference)."""
    mode = p.get("mode", "semantic")
    rel = p["rel"]
    ref_pose = _ref_pose(s, p)
    if mode == "absolute" and p.get("abs_target") is not None:
        return p["abs_target"]
    if mode == "relative" or p.get("relation") is None or p.get("reference", "world") == "world":
        return apply(ref_pose, rel)
    # semantic
    R = p["relation"]
    support = s.objects.get(p["reference"])
    subj = s.objects.get(p["object"])
    if support is None or subj is None:
        return apply(ref_pose, rel)
    yaw = support.pose[3]
    c, sn = np.cos(yaw), np.sin(yaw)
    if R["type"] == "on_top":
        ax, ay = R["align_xy"]
        wx = support.pose[0] + c * ax - sn * ay
        wy = support.pose[1] + sn * ax + c * ay
        z = support.pose[2] + support.size[2] / 2 + subj.size[2] / 2 + R.get("clearance", 0.004)
        return pose(wx, wy, z, yaw)
    if R["type"] == "beside":
        dx, dy = R["dir"]
        dist = R["dist_norm"] * (subj.size[0] / 2 + support.size[0] / 2)
        wx = support.pose[0] + (c * dx - sn * dy) * dist
        wy = support.pose[1] + (sn * dx + c * dy) * dist
        return pose(wx, wy, s.table_z + subj.size[2] / 2, yaw)
    return apply(ref_pose, rel)


def _place_act(s: BeliefState, p):
    """Carry the grasped subject to a target pose, then descend and release.
    `lift` sets travel height (caution knob the imagined search varies)."""
    target = _place_target(s, p)
    g = s.gripper
    lift = p.get("lift", APPROACH_Z)
    over = pose(target[0], target[1], max(target[2] + lift, g[2]), target[3])
    if dist_xy(g, over) > 0.02:
        return Action(target=over, gripper_close=1.0)      # travel at height
    descended = abs(g[2] - target[2]) < 0.015
    return Action(target=pose(target[0], target[1], target[2], target[3]),
                  gripper_close=0.0 if descended else 1.0)  # release when down

def _place_done(s, p):
    target = _place_target(s, p)
    subj = s.objects[p["object"]]
    return s.grasped != p["object"] and dist_xyz(subj.pose, target) < 0.03


def _move_act(s: BeliefState, p):
    ref = _ref_pose(s, p)
    target = apply(ref, p["rel"])
    lift = p.get("lift", 0.0)
    g = s.gripper
    if lift > 0 and dist_xy(g, target) > 0.02:
        return Action(target=pose(target[0], target[1], target[2] + lift, target[3]),
                      gripper_close=1.0)
    return Action(target=pose(target[0], target[1], target[2], target[3]),
                  gripper_close=1.0)

def _move_done(s, p):
    ref = _ref_pose(s, p)
    target = apply(ref, p["rel"])
    return dist_xyz(s.gripper, target) < 0.02


def _release_act(s, p):
    return Action(target=pose(s.gripper[0], s.gripper[1], s.gripper[2] + 0.03, s.gripper[3]),
                  gripper_close=0.0)

def _release_done(s, p):
    return s.grasped is None


def _ref_pose(s: BeliefState, p) -> Pose:
    ref = p.get("reference", "world")
    if ref == "world":
        return pose(0, 0, 0, 0)
    return s.objects[ref].pose


SKILL_LIBRARY: dict[str, Skill] = {
    "reach":   Skill("reach",   "above",   _reach_act,   _reach_done,   lambda s, p: True),
    "grasp":   Skill("grasp",   "grasped", _grasp_act,   _grasp_done,   lambda s, p: s.grasped is None),
    "move":    Skill("move",    "near",    _move_act,    _move_done,    lambda s, p: s.grasped == p["object"]),
    "place":   Skill("place",   "on_top",  _place_act,   _place_done,   lambda s, p: s.grasped == p["object"]),
    "release": Skill("release", "on_table", _release_act, _release_done, lambda s, p: True),
}
