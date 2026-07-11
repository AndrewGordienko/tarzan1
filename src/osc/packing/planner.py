"""Beam search over task-level packing decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools

from .candidates import candidate_placements
from .domain import PackingProgram, PackingState
from .world_model import apply_placement, remove_placement


@dataclass
class PackingAction:
    kind: str
    item_id: str
    placement: object = None
    removed: tuple[str, ...] = ()


@dataclass
class PackingPlan:
    actions: list[PackingAction]
    state: PackingState
    score: float
    feasible: bool
    rearranged: bool = False
    reason: str = ""


class PackingPlanner:
    def __init__(self, program: PackingProgram | None = None, beam_width=32):
        self.program = program or PackingProgram([], [], [], [])
        self.beam_width = beam_width

    def _score(self, state, actions, pending):
        packed = len(state.placements)
        future = sum(state.items[i].volume for i in pending if i not in state.placements)
        rehandles = sum(a.kind == "temporarily_remove" for a in actions)
        heavy_low = sum(state.items[i].mass * p.position[2] for i, p in state.placements.items())
        fragile_high = sum(p.position[2] for i, p in state.placements.items()
                           if state.items[i].fragile)
        return (packed * 100.0 - future * 0.01 - rehandles * 2.0
                - heavy_low * 0.05 + fragile_high * 0.02 + state.packed_volume * .001)

    def plan(self, state: PackingState, pending=None) -> PackingPlan:
        pending = [i for i in (pending or state.items) if i not in state.placements]
        beam = [(state, [], pending)]
        finished = []
        for _ in range(len(pending)):
            nxt = []
            for cur, actions, left in beam:
                if not left:
                    finished.append((cur, actions)); continue
                # Try every remaining item: heavy/large ordering emerges from score.
                for item_id in left:
                    item = cur.items[item_id]
                    candidates = list(candidate_placements(cur, item))
                    for p in candidates[:24]:
                        ns = apply_placement(cur, p)
                        na = actions + [PackingAction("place_inside", item_id, p)]
                        nl = [x for x in left if x != item_id]
                        nxt.append((ns, na, nl))
                if not nxt and left:
                    finished.append((cur, actions))
            nxt.sort(key=lambda x: self._score(x[0], x[1], x[2]), reverse=True)
            beam = nxt[:self.beam_width]
            if not beam:
                break
        finished.extend((s, a) for s, a, _ in beam if not _)
        if not finished:
            return PackingPlan([], state, -1e9, False, reason="no_candidate")
        best_state, best_actions = max(finished, key=lambda x: self._score(x[0], x[1], pending))
        feasible = all(i in best_state.placements for i in pending)
        return PackingPlan(best_actions, best_state, self._score(best_state, best_actions, pending),
                           feasible, False, "beam_search")

    def replan_with_rearrangement(self, state: PackingState, pending_item: str) -> PackingPlan:
        """Remove a minimum set of existing items, then repack all items jointly."""
        ids = list(state.placements)
        for k in range(1, len(ids) + 1):
            for removed in itertools.combinations(ids, k):
                base = state
                for item_id in removed:
                    base = remove_placement(base, item_id)
                plan = self.plan(base, pending=[pending_item] + [i for i in removed])
                if plan.feasible:
                    actions = [PackingAction("temporarily_remove", i) for i in removed] + plan.actions
                    plan.actions = actions
                    plan.rearranged = True
                    plan.state.history.append({"action": "rearrange", "removed": list(removed),
                                               "trigger": pending_item})
                    return plan
        return PackingPlan([], state, -1e9, False, True, "no_rearrangement_feasible")
