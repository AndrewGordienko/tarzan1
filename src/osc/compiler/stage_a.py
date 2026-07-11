"""Stage A: compile one demonstration into a TaskGraph.

Pipeline: keyframes -> per-segment "which object moved and relative to what" ->
predicates + relative transforms -> goal set. No weights are updated; this is
pure inference over the single demo, done once.
"""
from __future__ import annotations

import numpy as np

from ..geometry import Pose, dist_xy, relative
from ..perception.tracks import DemoTrace, Keyframe, segment_keyframes
from .task_graph import Predicate, TaskGraph, Transition

NEAR_XY = 0.06


def compile_demo(trace: DemoTrace, roles: dict[str, str]) -> TaskGraph:
    kfs = segment_keyframes(trace)
    objects = list(trace.object_tracks.keys())
    transitions: list[Transition] = []

    for a, b in zip(kfs[:-1], kfs[1:]):
        subject = _manipulated_object(a, b, objects)
        if subject is None:
            continue
        reference = _reference_frame(subject, b, objects, roles)
        subj_pose = b.object_poses[subject]
        ref_pose = _frame_pose(reference, b)
        rel = relative(ref_pose, subj_pose)
        add, remove = _predicate_delta(subject, reference, a, b, objects)
        transitions.append(Transition(
            subject=subject, reference=reference, rel_transform=rel,
            add=frozenset(add), remove=frozenset(remove),
            contact=(b.grasped == subject), reason=b.reason))

    goal = _goal_predicates(kfs[-1], objects, roles)
    return TaskGraph(transitions=_merge_trivial(transitions), goal=frozenset(goal),
                     objects=objects, roles=roles)


def _manipulated_object(a: Keyframe, b: Keyframe, objects: list[str]) -> str | None:
    """The object whose pose changed the most between two keyframes (or the one
    being grasped), i.e. the manipuland of this segment."""
    if b.grasped is not None:
        return b.grasped
    if a.grasped is not None:
        return a.grasped
    best, best_d = None, 1e-3
    for n in objects:
        d = float(np.linalg.norm(b.object_poses[n][:3] - a.object_poses[n][:3]))
        if d > best_d:
            best, best_d = n, d
    return best


def _reference_frame(subject: str, b: Keyframe, objects: list[str],
                     roles: dict[str, str]) -> str:
    """Pick the frame the subgoal is most naturally expressed in: the nearest
    other object (typically the target/support), else the world/table."""
    others = [n for n in objects if n != subject]
    if not others:
        return "world"
    nearest = min(others, key=lambda n: dist_xy(b.object_poses[subject], b.object_poses[n]))
    if dist_xy(b.object_poses[subject], b.object_poses[nearest]) < 3 * NEAR_XY:
        return nearest
    return "world"


def _frame_pose(reference: str, kf: Keyframe) -> Pose:
    if reference == "world":
        from ..geometry import pose
        return pose(0, 0, 0, 0)
    return kf.object_poses[reference]


def _predicate_delta(subject, reference, a: Keyframe, b: Keyframe, objects):
    add, remove = set(), set()
    if b.grasped == subject and a.grasped != subject:
        add.add(Predicate("grasped", (subject,)))
        remove.add(Predicate("on_table", (subject,)))
    if a.grasped == subject and b.grasped != subject:
        remove.add(Predicate("grasped", (subject,)))
        # released: either placed on a support or back on the table
        if reference != "world" and b.object_poses[subject][2] > b.object_poses[reference][2]:
            add.add(Predicate("on_top", (subject, reference)))
        else:
            add.add(Predicate("on_table", (subject,)))
    if reference != "world" and dist_xy(b.object_poses[subject], b.object_poses[reference]) < NEAR_XY:
        add.add(Predicate("near", (subject, reference)))
    return add, remove


def _goal_predicates(last: Keyframe, objects, roles):
    """Goal = the durable spatial relations that hold at the end of the demo."""
    goal = set()
    for n in objects:
        p = last.object_poses[n]
        support = None
        for m in objects:
            if m == n:
                continue
            if dist_xy(p, last.object_poses[m]) < NEAR_XY and p[2] > last.object_poses[m][2] + 1e-3:
                support = m
        if support is not None:
            goal.add(Predicate("on_top", (n, support)))
    return goal


def _merge_trivial(transitions: list[Transition]) -> list[Transition]:
    """Keep only transitions that carry task meaning: a contact change, a durable
    predicate (grasped / on_table / on_top), or a removal. `near`-only, no-contact
    transitions are perception jitter after the object is already placed and are
    dropped (the durable `near` relations still surface in the goal set)."""
    out = []
    for tr in transitions:
        durable = {p for p in tr.add if p.name != "near"}
        if tr.contact or durable or tr.remove:
            out.append(tr)
    return out
