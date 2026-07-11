"""Stage B: ground the task graph onto skill experts (the sparse router).

For each transition in the compiled program, retrieve the small set of skills
whose effects realize that subgoal and retarget them to the *current* objects
and poses by filling in params (object, reference frame, relative transform).
This is the router: from a whole library, only the handful of experts needed for
the current phase are instantiated. Retargeting uses the relative transform from
the demo, so a subgoal recorded once transfers to any layout.
"""
from __future__ import annotations

from .library import SKILL_LIBRARY, SkillInstance
from ..compiler.task_graph import TaskGraph, Transition


def ground_transition(tr: Transition) -> list[SkillInstance]:
    """Expand one subgoal into an ordered micro-sequence of grounded skills."""
    subj, ref, rel = tr.subject, tr.reference, tr.rel_transform
    seq: list[SkillInstance] = []
    if tr.contact:
        # subject must be in-hand for this transition: acquire if needed, then
        # transport to the relative target. A pure grasp-acquisition (target is
        # the object's own spot, reference == world) needs no transport.
        seq.append(SkillInstance(SKILL_LIBRARY["reach"], {"object": subj}, f"reach:{subj}"))
        seq.append(SkillInstance(SKILL_LIBRARY["grasp"], {"object": subj}, f"grasp:{subj}"))
        if ref != "world":
            seq.append(SkillInstance(SKILL_LIBRARY["move"],
                                     {"object": subj, "reference": ref, "rel": rel},
                                     f"move:{subj}->{ref}"))
    ends_released = any(p.name in ("on_table", "on_top") for p in tr.add)
    if ends_released:
        seq.append(SkillInstance(SKILL_LIBRARY["place"],
                                 {"object": subj, "reference": ref, "rel": rel},
                                 f"place:{subj}@{ref}"))
    return seq


def ground_plan(graph: TaskGraph) -> list[SkillInstance]:
    """The nominal grounded plan: concatenate each transition's micro-sequence,
    collapsing redundant re-grasps of an already-held object."""
    plan: list[SkillInstance] = []
    held = None
    for tr in graph.transitions:
        for si in ground_transition(tr):
            if si.skill.name in ("reach", "grasp") and held == si.params.get("object"):
                continue                        # already in hand; skip re-grasp
            plan.append(si)
            if si.skill.name == "grasp":
                held = si.params["object"]
            if si.skill.name in ("place", "release"):
                held = None
    return plan
