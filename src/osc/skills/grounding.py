"""Stage B: ground the role-based task graph onto skill experts (the router).

Takes a correspondence mapping (role -> eval track id) and expands each
transition into an ordered micro-sequence of grounded skills whose params refer
to concrete track ids. Only the handful of experts a phase needs are
instantiated.
"""
from __future__ import annotations

from .library import SKILL_LIBRARY, SkillInstance
from ..compiler.task_graph import TaskGraph, Transition


def _resolve(role: str, corr: dict) -> str:
    return "world" if role == "world" else corr.get(role, role)


def ground_transition(tr: Transition, corr: dict, mode: str = "semantic"):
    subj = _resolve(tr.subject, corr)
    ref = _resolve(tr.reference, corr)
    rel = tr.rel_transform
    seq = []
    if subj is None:
        return seq
    if tr.contact:
        seq.append(SkillInstance(SKILL_LIBRARY["reach"], {"object": subj}, f"reach:{subj}"))
        seq.append(SkillInstance(SKILL_LIBRARY["grasp"], {"object": subj}, f"grasp:{subj}"))
        if ref != "world":
            seq.append(SkillInstance(SKILL_LIBRARY["move"],
                                     {"object": subj, "reference": ref, "rel": rel},
                                     f"move:{subj}->{ref}"))
    if any(p.name in ("on_table", "on_top", "at_rel") for p in tr.add):
        seq.append(SkillInstance(SKILL_LIBRARY["place"],
                                 {"object": subj, "reference": ref, "rel": rel,
                                  "relation": tr.relation, "abs_target": tr.abs_target,
                                  "mode": mode},
                                 f"place:{subj}@{ref}"))
    return seq


def ground_plan(graph: TaskGraph, corr: dict, mode: str = "semantic"):
    plan, held = [], None
    for tr in graph.transitions:
        for si in ground_transition(tr, corr, mode):
            if si.skill.name in ("reach", "grasp") and held == si.params.get("object"):
                continue
            plan.append(si)
            if si.skill.name == "grasp":
                held = si.params["object"]
            if si.skill.name in ("place", "release"):
                held = None
    return plan


def ground_goal(graph: TaskGraph, corr: dict) -> frozenset:
    """Rewrite the goal predicates from roles to concrete track ids."""
    from ..compiler.task_graph import Predicate
    out = set()
    for p in graph.goal:
        args = tuple(_resolve(a, corr) for a in p.args)
        out.add(Predicate(p.name, args))
    return frozenset(out)


def ground_goal_rel(graph: TaskGraph, corr: dict) -> dict:
    """Resolve the expected-offset map (for at_rel goals) to track ids."""
    return {(_resolve(a, corr), _resolve(b, corr)): rel
            for (a, b), rel in graph.goal_rel.items()}
