"""Task registry: scenes (ground-truth roles), scripted oracles that record the
single demo, and ground-truth success predicates.

Ground-truth objects are named by role for the scorer; the agent only ever sees
nameless percepts, so this leaks nothing. record_demo drives the oracle in a
clean scene but the compiler sees the demo through perception (Corruptor +
StateEstimator), exactly like execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .agent.estimator import StateEstimator
from .compiler.stage_a import compile_demo
from .compiler.task_graph import TaskGraph
from .geometry import pose
from .perception.detections import Corruptor, CorruptionSpec
from .perception.tracks import extract_tracks
from .sim.base import Action
from .sim.randomize import nominal
from .benchmark.scorer import gt_on_top, gt_on_table, gt_near
from .geometry import dist_xy as _dxy


def _beside(s, a, b, lo=0.05, hi=0.16):
    """a on the table, offset from b within the demonstrated side-placement band
    (not stacked, not on top)."""
    if a in s.fallen or b in s.fallen:
        return False
    return gt_on_table(s, a) and lo < _dxy(s.objects[a].pose, s.objects[b].pose) < hi


@dataclass
class TaskSpec:
    name: str
    scene: dict
    oracle: Callable
    success: Callable
    max_total_steps: int = 400


# ---------------------------------------------------------------- oracles
def _pick(a, z_lift=0.10):
    return [(pose(a[0], a[1], a[2] + 0.09, 0.0), 0.0),
            (pose(a[0], a[1], a[2], 0.0), 0.0),
            (pose(a[0], a[1], a[2], 0.0), 1.0),
            (pose(a[0], a[1], a[2] + z_lift, 0.0), 1.0)]

def _place_on(a, b, sizes):
    top = b[2] + sizes[1] / 2 + sizes[0] / 2
    return [(pose(b[0], b[1], a[2] + 0.12, 0.0), 1.0),
            (pose(b[0], b[1], top, 0.0), 1.0),
            (pose(b[0], b[1], top, 0.0), 0.0),
            (pose(b[0], b[1], top + 0.09, 0.0), 0.0)]

def _place_beside(a, b, sizes, dx=0.10):
    tx, ty, tz = b[0] + dx, b[1], sizes[0] / 2
    return [(pose(tx, ty, a[2] + 0.12, 0.0), 1.0),
            (pose(tx, ty, tz, 0.0), 1.0),
            (pose(tx, ty, tz, 0.0), 0.0),
            (pose(tx, ty, tz + 0.09, 0.0), 0.0)]


def stack_oracle(s):
    a, b = s.objects["manip"].pose, s.objects["target"].pose
    sizes = (s.objects["manip"].size[2], s.objects["target"].size[2])
    return _pick(a) + _place_on(a, b, sizes)

def side_oracle(s):
    a, b = s.objects["manip"].pose, s.objects["target"].pose
    sizes = (s.objects["manip"].size[2], s.objects["target"].size[2])
    return _pick(a) + _place_beside(a, b, sizes)

def double_oracle(s):
    a, b = s.objects["manip"].pose, s.objects["target"].pose
    a2 = s.objects["manip2"].pose
    sa = (s.objects["manip"].size[2], s.objects["target"].size[2])
    seq = _pick(a) + _place_on(a, b, sa)
    # after first stack, manip sits on target; put manip2 on top of manip
    top1 = b[2] + s.objects["target"].size[2] / 2 + s.objects["manip"].size[2]
    stacked_manip = pose(b[0], b[1], top1 - s.objects["manip"].size[2] / 2, 0.0)
    sb = (s.objects["manip2"].size[2], s.objects["manip"].size[2])
    return seq + _pick(a2) + _place_on(a2, stacked_manip, sb)


# ---------------------------------------------------------------- scenes
def _obj(name, role, x, y, shape="box", size=0.04):
    return {"name": name, "role": role, "base_pose": [x, y, 0.02, 0.0],
            "shape": shape, "size": size}

STACK = TaskSpec(
    name="stack",
    scene={"table_bounds": (-0.3, 0.3, -0.3, 0.3),
           "objects": [_obj("manip", "manipuland", -0.08, 0.05, size=0.036),
                       _obj("target", "target", 0.10, -0.03, size=0.05)]},
    oracle=stack_oracle,
    success=lambda s, roles: gt_on_top(s, "manip", "target"))

SIDE_PLACE = TaskSpec(
    name="side_place",
    scene={"table_bounds": (-0.3, 0.3, -0.3, 0.3),
           "objects": [_obj("manip", "manipuland", -0.08, 0.05, size=0.036),
                       _obj("target", "target", 0.08, -0.03, size=0.05)]},
    oracle=side_oracle,
    success=lambda s, roles: _beside(s, "manip", "target"))

DOUBLE_STACK = TaskSpec(
    name="double_stack",
    scene={"table_bounds": (-0.3, 0.3, -0.3, 0.3),
           "objects": [_obj("manip", "manipuland", -0.10, 0.06, size=0.036),
                       _obj("manip2", "manipuland2", -0.02, 0.12, size=0.030),
                       _obj("target", "target", 0.10, -0.04, size=0.055)]},
    oracle=double_oracle,
    success=lambda s, roles: gt_on_top(s, "manip", "target") and gt_on_top(s, "manip2", "manip"),
    max_total_steps=700)

TASKS = {t.name: t for t in (STACK, SIDE_PLACE, DOUBLE_STACK)}

# The default benchmark uses the two robust single-manipuland tasks. DOUBLE_STACK
# compiles correctly (multi-episode Stage A) but simultaneous two-object track
# management at execution time is not yet reliable, so it is excluded from the
# headline numbers and kept as an experimental/known-limitation task.
DEFAULT_TASKS = [STACK, SIDE_PLACE]


# ---------------------------------------------------------------- demo
def record_demo(task: TaskSpec, settle_steps: int = 6) -> TaskGraph:
    state, backend, _ = nominal(task.scene)
    backend.reset(state)
    est = StateEstimator()
    corr = Corruptor(CorruptionSpec(pos_noise=0.002, occlusion_prob=0.0, drop_prob=0.0,
                                    delay_frames=0, false_contact_prob=0.0,
                                    identity_swap_prob=0.0),
                     np.random.default_rng(0))
    beliefs = [est.update(corr(backend.perceive()))]
    for target, grip in task.oracle(backend.state()):
        for _ in range(settle_steps):
            backend.step(Action(target=target, gripper_close=grip))
            beliefs.append(est.update(corr(backend.perceive())))
    graph = compile_demo(extract_tracks(beliefs))
    graph.role_to_gt = _role_to_gt(graph, beliefs[-1], backend.state())
    return graph


def _role_to_gt(graph, final_belief, gt_state) -> dict:
    """Map each compiled role to the ground-truth object name it corresponds to,
    by matching the role's demo track position to the nearest GT object. Used ONLY
    by oracle attribution modes and scoring -- never by the deployed agent."""
    from .geometry import dist_xyz
    mapping = {}
    for role, tid in graph.demo_role_tracks.items():
        if tid not in final_belief.objects:
            continue
        p = final_belief.objects[tid].pose
        best = min(gt_state.objects, key=lambda n: dist_xyz(gt_state.objects[n].pose, p))
        mapping[role] = best
    return mapping
