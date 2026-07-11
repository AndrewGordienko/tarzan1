"""Probabilistic, sticky, relational role<->track correspondence.

The v0.2 greedy matcher bound each role to its individually best-matching track,
which the attribution ladder showed was the dominant failure (misbound in ~62% of
episodes). RoleBelief fixes the modelling error:

  * GLOBAL one-to-one assignment (enumerate injective role->track maps; small),
    scored jointly -- not greedy per role,
  * RELATIONAL cost: the demonstrated *size ratio* between roles (e.g. manipuland
    smaller than support) must be preserved, which disambiguates similarly-sized
    distractors that fool absolute-size matching,
  * a NULL option so a role with no good track is left unbound rather than forced,
  * CONFIDENCE + entropy from the softmax over assignment costs, and an AMBIGUOUS
    flag when the top two assignments are within a margin -- these drive active
    perception and honest "ambiguous episode" labelling,
  * STICKINESS: keep the previous assignment across replans unless new evidence is
    clearly better, so bindings don't flip every planning call.

Deterministic throughout (sorted iteration, deterministic color code).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ..agent.belief import BeliefState

# size_x, size_z, shape, color. Color weight is 0: demo and eval appearance are
# INDEPENDENTLY randomized, so a demo colour carries no information about which
# eval object plays a role -- weighting it only injects noise that mis-binds.
W = np.array([3.0, 3.0, 1.0, 0.0])
W_RATIO = 2.0                           # weight on demonstrated size-ratio consistency
STICK = 0.15                            # keep previous unless improvement exceeds this
AMBIG_MARGIN = 0.06                     # top-2 cost gap below this => genuinely near-tied
SOFTMAX_SCALE = 0.25
ROLE_CONF_MIN = 0.55                    # weakest-role marginal below this => ambiguous
ROLE_MARGIN_MIN = 0.20                  # weakest-role chosen-vs-runnerup gap => ambiguous


@dataclass
class RoleAssignment:
    mapping: dict                       # role -> track_id (or absent if null)
    confidence: float                   # 0..1 from softmax margin
    entropy: float
    ambiguous: bool
    per_role_conf: dict = field(default_factory=dict)
    margin: float = 1.0                 # weakest-role chosen-vs-runnerup posterior gap
    assignment_margin: float = 1.0      # top-1 vs top-2 joint-assignment cost gap
    top_cost: float = 0.0               # best joint-assignment cost
    second_cost: float = 0.0            # runner-up joint-assignment cost
    n_candidates: int = 0               # number of tracks considered


class RoleBelief:
    def __init__(self, role_signatures: dict):
        self.sigs = {r: np.asarray(s, dtype=float)
                     for r, s in role_signatures.items() if s is not None}
        self.roles = sorted(self.sigs)
        # demonstrated size ratios between role pairs (invariant to global scale)
        self._demo_ratio = {}
        for a, b in itertools.combinations(self.roles, 2):
            self._demo_ratio[(a, b)] = self.sigs[a][0] / max(1e-6, self.sigs[b][0])
        self.prev: dict | None = None

    def _assignment_cost(self, assign: dict, feats: dict) -> float:
        c = 0.0
        for role, tid in assign.items():
            c += float(np.linalg.norm(W * (self.sigs[role] - feats[tid])))
        for (a, b), dr in self._demo_ratio.items():
            if a in assign and b in assign:
                ar = feats[assign[a]][0] / max(1e-6, feats[assign[b]][0])
                c += W_RATIO * abs(math.log(max(1e-6, ar) / max(1e-6, dr)))
        return c

    def update(self, belief: BeliefState, fixed: dict | None = None) -> RoleAssignment:
        """`fixed` pins clarified roles to tracks; the one-to-one assignment is then
        recomputed over the REMAINING roles/tracks, so pinning one role propagates
        (a used track is removed from the others) and every role is re-scored under
        the constraint. Clarification is a global transaction, not a local override."""
        fixed = {r: t for r, t in (fixed or {}).items()
                 if r in self.sigs and t in belief.objects}
        tracks = sorted(belief.objects)
        feats = {t: belief.objects[t].feature() for t in tracks}
        free_roles = [r for r in self.roles if r not in fixed]
        available = [t for t in tracks if t not in set(fixed.values())]
        if not self.roles or len(available) < len(free_roles):
            base = dict(self.prev or {}); base.update(fixed)
            pr = {r: (1.0 if r in fixed else 0.0) for r in self.roles}
            return RoleAssignment(base, 0.0, 0.0, True, pr)

        scored = []
        if free_roles:
            for combo in itertools.permutations(available, len(free_roles)):
                assign = dict(fixed); assign.update(zip(free_roles, combo))
                scored.append((self._assignment_cost(assign, feats), assign))
        else:
            scored.append((self._assignment_cost(dict(fixed), feats), dict(fixed)))
        scored.sort(key=lambda x: x[0])
        best_cost, best = scored[0]
        second_cost = scored[1][0] if len(scored) > 1 else best_cost + 10.0
        margin = second_cost - best_cost

        # softmax posterior over assignments -> confidence + entropy
        costs = np.array([c for c, _ in scored])
        p = np.exp(-(costs - best_cost) / SOFTMAX_SCALE)
        p /= p.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        # stickiness among FREE roles only; a fixed (clarified) role is authoritative.
        chosen = best
        if not fixed and self.prev is not None and all(r in self.prev for r in self.roles):
            if all(self.prev[r] in tracks for r in self.roles):
                prev_cost = self._assignment_cost({r: self.prev[r] for r in self.roles}, feats)
                if prev_cost - best_cost < STICK:
                    chosen = {r: self.prev[r] for r in self.roles}
        self.prev = dict(chosen)

        # PER-ROLE marginal posterior: mass on each role's CHOSEN track, summed
        # over every (constrained) assignment that pairs them. Fixed roles are
        # certain by construction.
        per_role_conf = {}
        per_role_margin = {}
        for r in self.roles:
            if r in fixed:
                per_role_conf[r] = 1.0; per_role_margin[r] = 1.0
                continue
            mass = {}
            for prob, assign in zip(p, (a for _, a in scored)):
                mass[assign[r]] = mass.get(assign[r], 0.0) + float(prob)
            per_role_conf[r] = mass.get(chosen[r], 0.0)
            ranked = sorted(mass.values(), reverse=True)
            per_role_margin[r] = (ranked[0] - ranked[1]) if len(ranked) > 1 else ranked[0]
        # weakest-link confidence, and ambiguous if any role is contested.
        confidence = float(min(per_role_conf.values())) if per_role_conf else 0.0
        weakest_margin = min(per_role_margin.values()) if per_role_margin else 1.0
        ambiguous = (margin < AMBIG_MARGIN) or (confidence < ROLE_CONF_MIN) \
            or (weakest_margin < ROLE_MARGIN_MIN)
        return RoleAssignment(chosen, confidence, entropy, ambiguous, per_role_conf,
                              margin=weakest_margin, assignment_margin=margin,
                              top_cost=best_cost, second_cost=second_cost,
                              n_candidates=len(tracks))


def correspond(belief: BeliefState, role_signatures: dict) -> dict:
    """One-shot convenience wrapper (no stickiness). Returns just the mapping."""
    return RoleBelief(role_signatures).update(belief).mapping
