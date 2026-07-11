"""Uncertainty-aware global role<->track correspondence.

The v0.2 greedy matcher bound each role to its individually best-matching track,
which the attribution ladder showed was the dominant failure (misbound in ~62% of
episodes). RoleBelief fixes the modelling error:

  * GLOBAL one-to-one assignment (enumerate injective role->track maps; small),
    scored jointly -- not greedy per role,
  * a covariance-aware Gaussian negative log likelihood over every observed
    feature dimension, combining demonstration and deployment covariance,
  * explicit NULL assignments so a role with no supported candidate is not forced,
  * CONFIDENCE + entropy from the posterior over complete assignments, and an AMBIGUOUS
    flag when the top two assignments are within a margin -- these drive active
    perception and honest "ambiguous episode" labelling,
Deterministic throughout (sorted iteration, deterministic color code).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..agent.belief import BeliefState

# feature dimensions: size_x, size_z, shape, color.  Colour is deliberately
# excluded: demo/eval colours are independently randomized.  Shape is observed
# categorically with a small nominal variance; size dimensions participate only
# while their calibrated covariance says they are observable.
OBSERVED_DIMS = (0, 1, 2, 4)
MAX_UNOBSERVED_VAR = 0.05 ** 2
VAR_FLOOR = 1e-8
NULL_COST = 10.0
TOP_K = 256
AMBIG_MARGIN = 0.06                     # top-2 cost gap below this => genuinely near-tied
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
    association_contested: tuple = ()   # roles whose chosen track has contested identity
    posterior: list = field(default_factory=list)  # complete assignments: {mapping,cost,prob}
    null_mass: float = 0.0
    posterior_mass_outside_top_k: float = 0.0
    observed_dimensions: dict = field(default_factory=dict)


class RoleBelief:
    def __init__(self, role_signatures: dict, role_signature_vars: dict | None = None,
                 top_k: int | None = None, model_error_var: float = 1e-6,
                 allow_null: bool = True, normalize_dimensions: bool = False,
                 covariance_floor: float = VAR_FLOOR):
        self.sigs = {r: np.asarray(s, dtype=float)
                     for r, s in role_signatures.items() if s is not None}
        provided = role_signature_vars or {}
        self.sig_vars = {r: np.asarray(provided.get(r, np.full_like(s, 0.02 ** 2)), dtype=float)
                         for r, s in self.sigs.items()}
        # Categorical shape is observed exactly in this toy frontend.
        for v in self.sig_vars.values():
            if len(v) > 2:
                v[2] = min(v[2], 1e-6)
        self.roles = sorted(self.sigs)
        self.prev: dict | None = None
        self.top_k = top_k
        self.model_error_var = float(model_error_var)
        self.allow_null = bool(allow_null)
        self.normalize_dimensions = bool(normalize_dimensions)
        self.covariance_floor = float(covariance_floor)

    def _pair_cost(self, role: str, tid: str, feats: dict, feat_vars: dict) -> float:
        """Gaussian NLL for the dimensions both demo and deployment observed.

        C(r,t) = sum_d ((x_r-x_t)^2 / (var_r+var_t) + log(var_r+var_t)).
        A large per-axis camera variance excludes an occluded dimension instead
        of treating its neutral placeholder as evidence.
        """
        x_r, x_t = self.sigs[role], feats[tid]
        v_r, v_t = self.sig_vars[role], feat_vars[tid]
        c = 0.0
        n = 0
        for d in OBSERVED_DIMS:
            if d >= len(x_r) or d >= len(x_t):
                continue
            # Shape's nominal covariance is intentionally small; size axes are
            # omitted whenever either side says that view did not observe them.
            if d < 2 and (v_r[d] > MAX_UNOBSERVED_VAR or v_t[d] > MAX_UNOBSERVED_VAR):
                continue
            v = max(self.covariance_floor, float(v_r[d] + v_t[d] + self.model_error_var))
            c += float((x_r[d] - x_t[d]) ** 2 / v + math.log(v))
            n += 1
        # No shared observed attribute is unsupported evidence, not a free match.
        return (c / n if self.normalize_dimensions and n else c) if n else NULL_COST

    def _assignment_cost(self, assign: dict, feats: dict, feat_vars: dict) -> float:
        c = 0.0
        for role in self.roles:
            tid = assign.get(role)
            c += NULL_COST if tid is None else self._pair_cost(role, tid, feats, feat_vars)
        return c

    @staticmethod
    def _complete_assignments(roles, tracks, allow_null=True):
        """Enumerate one-to-one complete assignments with an explicit NULL."""
        def visit(i, used, current):
            if i == len(roles):
                yield dict(current); return
            role = roles[i]
            if allow_null:
                current[role] = None
                yield from visit(i + 1, used, current)
            for tid in tracks:
                if tid not in used:
                    current[role] = tid
                    used.add(tid)
                    yield from visit(i + 1, used, current)
                    used.remove(tid)
            current.pop(role, None)
        yield from visit(0, set(), {})

    def update(self, belief: BeliefState, fixed: dict | None = None) -> RoleAssignment:
        """`fixed` pins clarified roles to tracks; the one-to-one assignment is then
        recomputed over the REMAINING roles/tracks, so pinning one role propagates
        (a used track is removed from the others) and every role is re-scored under
        the constraint. Clarification is a global transaction, not a local override."""
        fixed = {r: t for r, t in (fixed or {}).items()
                 if r in self.sigs and t in belief.objects}
        tracks = sorted(belief.objects)
        feats = {t: belief.objects[t].feature() for t in tracks}
        feat_vars = {t: belief.objects[t].feature_var() for t in tracks}
        free_roles = [r for r in self.roles if r not in fixed]
        available = [t for t in tracks if t not in set(fixed.values())]
        if not self.roles:
            base = dict(self.prev or {}); base.update(fixed)
            pr = {r: (1.0 if r in fixed else 0.0) for r in self.roles}
            return RoleAssignment(base, 0.0, 0.0, True, pr)

        scored = []
        for partial in self._complete_assignments(free_roles, available, self.allow_null):
            assign = dict(fixed); assign.update(partial)
            scored.append((self._assignment_cost(assign, feats, feat_vars), assign))
        if not scored:
            # With nulls disabled, a temporarily under-detected scene has no
            # legal complete assignment.  Represent that state explicitly as an
            # unsupported/null assignment so policy abstention is safe instead of
            # crashing the loop.
            unsupported = dict(fixed)
            unsupported.update({r: None for r in free_roles})
            return RoleAssignment(unsupported, 0.0, 0.0, True,
                                  {r: (1.0 if r in fixed else 0.0) for r in self.roles},
                                  n_candidates=len(tracks),
                                  posterior=[dict(mapping=unsupported, cost=NULL_COST * len(free_roles), prob=1.0)],
                                  null_mass=1.0,
                                  observed_dimensions={r: () for r in free_roles})
        scored.sort(key=lambda x: x[0])
        exact_scored = scored
        best_cost, best = exact_scored[0]
        second_cost = exact_scored[1][0] if len(exact_scored) > 1 else best_cost + 10.0
        margin = second_cost - best_cost

        # softmax posterior over assignments -> confidence + entropy
        exact_costs = np.array([c for c, _ in exact_scored])
        exact_p = np.exp(-(exact_costs - best_cost))
        exact_p /= exact_p.sum()
        outside = 0.0
        scored = exact_scored
        if self.top_k is not None and len(scored) > self.top_k:
            outside = float(exact_p[self.top_k:].sum())
            scored = scored[:self.top_k]
        costs = np.array([c for c, _ in scored])
        p = np.exp(-(costs - best_cost))
        p /= p.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))

        chosen = best
        self.prev = {r: t for r, t in chosen.items() if t is not None}

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
                mass[assign.get(r)] = mass.get(assign.get(r), 0.0) + float(prob)
            per_role_conf[r] = mass.get(chosen.get(r), 0.0) if chosen.get(r) is not None else 0.0
            ranked = sorted(mass.values(), reverse=True)
            per_role_margin[r] = (ranked[0] - ranked[1]) if len(ranked) > 1 else ranked[0]
        # weakest-link confidence, and ambiguous if any role is contested.
        confidence = float(min(per_role_conf.values())) if per_role_conf else 0.0
        weakest_margin = min(per_role_margin.values()) if per_role_margin else 1.0
        contested = tuple(sorted(r for r, t in chosen.items()
                                 if t is not None and getattr(belief.objects[t], "association_contested", False)))
        posterior = [dict(mapping=dict(a), cost=float(c), prob=float(prob))
                     for (c, a), prob in zip(scored, p)]
        exact_null_mass = float(sum(prob for item, prob in zip(exact_scored, exact_p)
                                    if any(t is None for t in item[1].values())))
        observed = {}
        for r, tid in chosen.items():
            if tid is None:
                observed[r] = ()
                continue
            xr, xt = self.sigs[r], feats[tid]
            vr, vt = self.sig_vars[r], feat_vars[tid]
            observed[r] = tuple(d for d in OBSERVED_DIMS if d < len(xr) and d < len(xt)
                                and (d >= 2 or (vr[d] <= MAX_UNOBSERVED_VAR and
                                                vt[d] <= MAX_UNOBSERVED_VAR)))
        ambiguous = bool(contested) or (margin < AMBIG_MARGIN) or (confidence < ROLE_CONF_MIN) \
            or (weakest_margin < ROLE_MARGIN_MIN)
        return RoleAssignment(chosen, confidence, entropy, ambiguous, per_role_conf,
                              margin=weakest_margin, assignment_margin=margin,
                              top_cost=best_cost, second_cost=second_cost,
                              n_candidates=len(tracks), association_contested=contested,
                              posterior=posterior,
                              null_mass=exact_null_mass,
                              posterior_mass_outside_top_k=outside,
                              observed_dimensions=observed)


def correspond(belief: BeliefState, role_signatures: dict, role_signature_vars: dict | None = None) -> dict:
    """One-shot convenience wrapper (no stickiness). Returns just the mapping."""
    return RoleBelief(role_signatures, role_signature_vars).update(belief).mapping
