"""Stage C: imagined search over skill plans, using the approximate PlanningModel.

Scores candidate plans by predicted success minus penalties for collision,
uncertainty, force and irreversible states, and returns the best. Candidate
generation currently varies travel height (caution); task-level alternatives
(reorder, obstruction-first, regrasp) are the documented next extension and plug
in here by returning structurally different plans.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..skills.library import SkillInstance
from .planning_model import PlanningModel, RolloutResult

W_IRREVERSIBLE = 5.0
W_FORCE = 2.0
W_COLLISION = 0.5
W_UNCERTAINTY = 1.0


@dataclass
class PlanScore:
    plan: list
    result: RolloutResult
    score: float

    def breakdown(self) -> str:
        r = self.result
        return (f"score={self.score:+.3f} P(succ)={r.success_prob:.2f} "
                f"coll={r.collision_risk:.2f} force={r.force_risk:.2f} "
                f"irrev={r.irreversible_risk:.2f} unc={r.uncertainty:.3f}")


class ImaginedSearch:
    def __init__(self, planning_model: PlanningModel):
        self.pm = planning_model

    def candidates(self, base_plan):
        variants = [base_plan]
        for lift in (0.06, 0.12, 0.18):
            variants.append([_with_lift(si, lift) for si in base_plan])
        return variants

    def select(self, belief, base_plan, goal_check) -> PlanScore:
        best = None
        for plan in self.candidates(base_plan):
            r = self.pm.rollout(belief, plan, goal_check)
            score = (r.success_prob - W_COLLISION * r.collision_risk
                     - W_FORCE * r.force_risk - W_IRREVERSIBLE * r.irreversible_risk
                     - W_UNCERTAINTY * r.uncertainty)
            cand = PlanScore(plan=plan, result=r, score=score)
            if best is None or cand.score > best.score:
                best = cand
        return best


def _with_lift(si: SkillInstance, lift: float) -> SkillInstance:
    if si.skill.name in ("move", "place"):
        params = dict(si.params)
        params["lift"] = lift
        return SkillInstance(si.skill, params, si.label)
    return si
