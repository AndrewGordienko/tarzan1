"""Closed-loop logical packing execution with reobserve/replan hooks."""
from __future__ import annotations

from .planner import PackingPlanner, PackingAction
from .verifier import verify_final_pack, verify_placement
from .world_model import apply_placement, remove_placement


class PackingExecutor:
    def __init__(self, planner: PackingPlanner):
        self.planner = planner

    def execute(self, state, arrivals):
        current = state.clone()
        log = []
        for item_id in arrivals:
            if item_id in current.placements:
                continue
            plan = self.planner.plan(current, pending=[item_id])
            if not plan.feasible:
                plan = self.planner.replan_with_rearrangement(current, item_id)
            if not plan.feasible:
                return current, log, False, "clarify_or_abstain: no feasible placement"
            for action in plan.actions:
                if action.kind == "temporarily_remove":
                    current = remove_placement(current, action.item_id)
                    log.append({"kind": action.kind, "item": action.item_id})
                elif action.placement is not None:
                    current = apply_placement(current, action.placement)
                    ok, detail = verify_placement(current, action.item_id)
                    log.append({"kind": action.kind, "item": action.item_id,
                                "verified": ok, "detail": detail,
                                "rearranged": plan.rearranged})
                    if not ok:
                        return current, log, False, "verification_failed"
        ok, detail = verify_final_pack(current, arrivals)
        return current, log, ok, "complete" if ok else "verification_failed"
