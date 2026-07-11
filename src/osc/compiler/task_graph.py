"""The object-centric task program: predicates + relative transforms.

A TaskGraph is what one demonstration compiles to. It is intentionally NOT a
trajectory: nodes are symbolic states (sets of predicates) and edges carry the
*relative transform* the manipulated object must reach w.r.t. a reference frame.
Because edges store relative transforms (not world poses), the same graph applies
unchanged when objects are moved, swapped or relit -- the invariance that makes
one-shot transfer possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose


@dataclass(frozen=True)
class Predicate:
    name: str                       # grasped | on_table | on_top | near | placed_at
    args: tuple                     # object names / reference frames

    def __str__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.args))})"


@dataclass
class Transition:
    """One edge of the program == one subgoal for one manipulated object."""
    subject: str                    # object being manipulated
    reference: str                  # frame the target is expressed relative to
    rel_transform: Pose             # subject pose in reference frame at the subgoal
    add: frozenset                  # predicates that become true
    remove: frozenset               # predicates that become false
    contact: bool                   # is the subject grasped during this transition
    reason: str = ""
    abs_target: object = None       # subject's absolute demo world pose (for the
                                    # relative-vs-absolute ablation only)

    def describe(self) -> str:
        d = f"move {self.subject} to rel{np.round(self.rel_transform, 3)} of {self.reference}"
        if self.add:
            d += "  =>  " + ", ".join(map(str, sorted(self.add, key=str)))
        return d


@dataclass
class TaskGraph:
    """A program over ROLES, not object names. `subject`/`reference` in each
    transition and in the goal are role labels (e.g. "manipuland", "target",
    "world"); `role_signatures` lets eval-time correspondence bind each role to a
    concrete estimator track by geometry/appearance."""
    transitions: list[Transition] = field(default_factory=list)
    goal: frozenset = frozenset()               # predicates over role labels
    goal_rel: dict = field(default_factory=dict)  # (subj_role,ref_role) -> expected rel
    roles: list[str] = field(default_factory=list)          # e.g. [manipuland, target]
    role_signatures: dict = field(default_factory=dict)     # role -> feature vector
    demo_role_tracks: dict = field(default_factory=dict)    # role -> demo track id (ref only)

    def pretty(self) -> str:
        lines = [f"TaskGraph  ({len(self.transitions)} transitions, roles={self.roles})"]
        for i, tr in enumerate(self.transitions):
            lines.append(f"  {i}. [{tr.reason}] {tr.describe()}")
        lines.append("  goal: " + ", ".join(map(str, sorted(self.goal, key=str))))
        return "\n".join(lines)
