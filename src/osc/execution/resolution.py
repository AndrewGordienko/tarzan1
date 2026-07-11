"""Ambiguity-resolution layer.

`RoleBelief` says *how sure* it is about each role. The `ResolutionPolicy` decides
what to DO when it isn't sure enough to commit -- kept as a distinct component so
its behaviour can be ablated separately from correspondence itself.

Two kinds of action solve DIFFERENT problems and are reported separately:

  * ACTIVE INSPECTION ("observe") resolves uncertainty caused by poor sensing --
    gather more frames so the estimator averages down pose noise / occlusion. It
    CANNOT resolve a genuine tie (two objects identical under every observation).
  * CLARIFICATION ("ask_user") introduces information that is NOT in the video --
    a user selection / SKU metadata. This is the ONLY thing that can resolve a
    fundamentally ambiguous scene.

The policy tries the cheapest safe action first (inspection), escalates to
clarification when inspection stops helping, and abstains rather than guessing.
A clarified role is remembered on the `TaskContext` and re-applied every planning
call, so the customer is asked once per workflow, not once per box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ResolutionKind = Literal["commit", "observe", "change_viewpoint", "probe",
                         "request_metadata", "ask_user", "abstain"]

# escalation order: cheapest/safest information-gathering first, human last, then
# refuse. Each physical action is tried at most once (a one-shot reveal); passive
# observation repeats until it stops paying off.
_PHYSICAL_ORDER = ("change_viewpoint", "probe", "request_metadata")


@dataclass
class ResolutionAction:
    kind: ResolutionKind
    target_roles: tuple = ()
    expected_information_gain: float = 0.0
    cost: float = 0.0
    risk: float = 0.0


@dataclass
class TaskContext:
    """Evidence the compiler may use beyond the demonstration itself. `instruction`
    / `object_metadata` are placeholders for the packing product; `user_selections`
    is the set of roles the customer has already disambiguated (persisted)."""
    instruction: str | None = None
    object_metadata: dict = field(default_factory=dict)
    user_selections: set = field(default_factory=set)   # roles already clarified
    hard_constraints: list = field(default_factory=list)
    soft_preferences: list = field(default_factory=list)


@dataclass
class ResolutionConfig:
    allow_inspection: bool = True
    allow_clarification: bool = True
    commit_threshold: float = 0.60   # weakest-role marginal needed to commit
    max_inspections: int = 3         # inspection *rounds* before escalating
    inspect_frames: int = 3          # frames gathered per inspection round
    min_inspect_gain: float = 0.02   # if a round improves weakest-conf less, stop
    max_clarifications: int = 2      # questions allowed per episode
    # optional physical actions (enabled per-scenario as the benchmark demands):
    allow_viewpoint: bool = False    # change_viewpoint: reveal an occluded feature
    allow_probe: bool = False        # probe/lift: reveal mass / graspability
    allow_metadata: bool = False     # request_metadata: SKU/order-system identity


class ResolutionPolicy:
    def __init__(self, cfg: ResolutionConfig | None = None):
        self.cfg = cfg or ResolutionConfig()

    def contested(self, ra, context: TaskContext) -> list:
        """Roles not yet trustworthy: below commit threshold and not user-resolved."""
        association_contested = set(getattr(ra, "association_contested", ()))
        return sorted(r for r, c in ra.per_role_conf.items()
                      if (c < self.cfg.commit_threshold or r in association_contested)
                      and r not in context.user_selections)

    def committable(self, ra, context: TaskContext) -> bool:
        return not self.contested(ra, context)

    def _enabled_physical(self):
        flags = {"change_viewpoint": self.cfg.allow_viewpoint,
                 "probe": self.cfg.allow_probe,
                 "request_metadata": self.cfg.allow_metadata}
        return [a for a in _PHYSICAL_ORDER if flags[a]]

    def decide(self, ra, context: TaskContext, inspections_used: int,
               clarifications_used: int, last_inspect_gain: float | None,
               tried_physical: set | None = None) -> ResolutionAction:
        tried_physical = tried_physical or set()
        contested = self.contested(ra, context)
        gain = self.cfg.commit_threshold - min((ra.per_role_conf[r] for r in contested), default=0.0)
        if not contested:
            return ResolutionAction("commit", risk=0.0)
        # 1) cheapest safe action: gather more passive observations -- but only
        #    while it is still paying off. A single unlucky round shouldn't abort;
        #    give inspection a short grace period before the diminishing-returns
        #    escalation kicks in.
        inspection_helping = (last_inspect_gain is None or inspections_used < 2
                              or last_inspect_gain >= self.cfg.min_inspect_gain)
        if (self.cfg.allow_inspection and inspections_used < self.cfg.max_inspections
                and inspection_helping):
            return ResolutionAction("observe", tuple(contested), gain,
                                    cost=float(self.cfg.inspect_frames))
        # 2) one-shot physical reveals (viewpoint -> probe -> metadata), each once.
        for kind in self._enabled_physical():
            if kind not in tried_physical:
                cost = {"change_viewpoint": 2.0, "probe": 4.0, "request_metadata": 0.5}[kind]
                risk = 0.2 if kind == "probe" else 0.0        # probing can disturb the scene
                return ResolutionAction(kind, tuple(contested), gain, cost=cost, risk=risk)
        # 3) human clarification -- ONE role at a time (highest value = lowest
        #    confidence); the one-to-one recompute may then resolve the others.
        if self.cfg.allow_clarification and clarifications_used < self.cfg.max_clarifications:
            worst = min(contested, key=lambda r: ra.per_role_conf[r])
            return ResolutionAction("ask_user", (worst,), gain, cost=1.0)
        # 4) never guess.
        return ResolutionAction("abstain", tuple(contested))
