"""Stage C: imagined search over skill sequences.

Generate a handful of candidate plans (the grounded nominal plan plus caution /
ordering variants), roll each through the world model, and pick the plan that
maximizes success while penalizing collision risk, uncertainty, force violations
and irreversible states. This is the "imagine before you act" step: alternatives
are tried in the head, not on the robot.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..skills.library import SkillInstance
from .model import RolloutResult, WorldModel

# cost weights: penalize irreversibility hardest, then force, collisions, doubt.
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
        return (f"score={self.score:+.3f}  P(success)={r.success_prob:.2f}  "
                f"collision={r.collision_risk:.2f}  force={r.force_risk:.2f}  "
                f"irrev={r.irreversible_risk:.2f}  unc={r.uncertainty:.3f}")


class ImaginedSearch:
    def __init__(self, world_model: WorldModel):
        self.wm = world_model

    def candidates(self, base_plan: list[SkillInstance]) -> list[list[SkillInstance]]:
        """Derive plan variants that differ in ways the world model can score:
        travel height (caution) is the main knob here."""
        variants = [base_plan]
        for lift in (0.06, 0.12, 0.18):
            variants.append([_with_lift(si, lift) for si in base_plan])
        return variants

    def select(self, state, base_plan, goal_check) -> PlanScore:
        best: PlanScore | None = None
        for plan in self.candidates(base_plan):
            r = self.wm.rollout(state, plan, goal_check)
            score = (r.success_prob
                     - W_COLLISION * r.collision_risk
                     - W_FORCE * r.force_risk
                     - W_IRREVERSIBLE * r.irreversible_risk
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
