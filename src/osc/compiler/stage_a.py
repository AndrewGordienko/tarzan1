"""Stage A: compile one demonstration (belief trajectory) into a role-based
TaskGraph, supporting MULTIPLE grasp episodes (compositional tasks).

Roles are inferred by function, never names:
  * each distinct grasped track -> a manipuland role (manipuland0, manipuland1...),
  * each placement reference -> that manipuland's role if it is one, else a
    support role (support0, support1, ...).
Each grasp episode yields a grasp transition and a place transition, expressed in
relative transforms so they transfer. Correspondence binds roles to eval tracks.
"""
from __future__ import annotations

import numpy as np

from ..geometry import pose, relative
from ..perception.tracks import DemoTrace
from .task_graph import Predicate, TaskGraph, Transition

NEAR_XY = 0.06


def _grasp_episodes(trace: DemoTrace):
    """Maximal spans where a single track is held. Returns [(track, t0, t1)]."""
    eps, cur, start = [], None, 0
    for t, g in enumerate(trace.grasped):
        if g != cur:
            if cur is not None:
                eps.append((cur, start, t - 1))
            cur, start = g, t
    if cur is not None:
        eps.append((cur, start, trace.T - 1))
    return eps


def _nearest_other(trace: DemoTrace, subject: str, t: int):
    """Placement reference. If the subject rests on something (an object below it
    at the same xy), that support is the reference -- this distinguishes stacked
    objects. Otherwise the nearest object in the plane (e.g. side placement)."""
    sp = trace.object_tracks[subject][t]
    supports = []
    for tid, track in trace.object_tracks.items():
        if tid == subject:
            continue
        p = track[t]
        if float(np.hypot(sp[0] - p[0], sp[1] - p[1])) < NEAR_XY and p[2] < sp[2] - 1e-3:
            supports.append((p[2], tid))            # object directly beneath
    if supports:
        return max(supports)[1]                     # the highest support below
    best, best_d = None, 3 * NEAR_XY
    for tid, track in trace.object_tracks.items():
        if tid == subject:
            continue
        d = float(np.hypot(sp[0] - track[t][0], sp[1] - track[t][1]))
        if d < best_d:
            best, best_d = tid, d
    return best


def compile_demo(trace: DemoTrace) -> TaskGraph:
    episodes = _grasp_episodes(trace)
    if not episodes:
        return TaskGraph()

    track_role: dict[str, str] = {}
    for tk, _, _ in episodes:
        if tk not in track_role:
            track_role[tk] = f"manipuland{len([r for r in track_role.values() if r.startswith('manipuland')])}"

    transitions, goal, goal_rel = [], set(), {}
    support_n = 0
    for tk, t0, t1 in episodes:
        subj = track_role[tk]
        # grasp transition (reference = world; reach/grasp use object's own pose)
        transitions.append(Transition(
            subject=subj, reference="world",
            rel_transform=relative(pose(), trace.object_tracks[tk][t0]),
            add=frozenset({Predicate("grasped", (subj,))}),
            remove=frozenset({Predicate("on_table", (subj,))}),
            contact=True, reason="grasp"))

        # placement transition (at release frame t1)
        ref_tk = _nearest_other(trace, tk, t1)
        if ref_tk is None:
            ref_role = "world"
        elif ref_tk in track_role:
            ref_role = track_role[ref_tk]
        else:
            ref_role = f"support{support_n}"; support_n += 1
            track_role[ref_tk] = ref_role

        subj_pose = trace.object_tracks[tk][t1]
        ref_pose = pose() if ref_role == "world" else trace.object_tracks[ref_tk][t1]
        rel = relative(ref_pose, subj_pose)
        stacked = ref_role != "world" and subj_pose[2] > ref_pose[2] + 1e-3
        if stacked:
            add = {Predicate("on_top", (subj, ref_role))}
        elif ref_role != "world":
            # placed on the table AT the demonstrated offset from a reference:
            # on_table alone is trivially already true, so the goal is the offset.
            add = {Predicate("at_rel", (subj, ref_role))}
            goal_rel[(subj, ref_role)] = rel
        else:
            add = {Predicate("on_table", (subj,))}
        transitions.append(Transition(
            subject=subj, reference=ref_role, rel_transform=rel,
            add=frozenset(add), remove=frozenset({Predicate("grasped", (subj,))}),
            contact=False, reason="place", abs_target=subj_pose.copy()))
        goal.update(add)

    sigs = {role: trace.features.get(tk) for tk, role in track_role.items()}
    return TaskGraph(transitions=transitions, goal=frozenset(goal), goal_rel=goal_rel,
                     roles=list(track_role.values()), role_signatures=sigs,
                     demo_role_tracks={r: t for t, r in track_role.items()})
