"""Packing completion and safety verification."""
from __future__ import annotations

from .domain import PackingState
from .world_model import evaluate_placement


def verify_placement(state, item_id):
    if item_id not in state.placements:
        return False, {"reason": "missing"}
    p = state.placements[item_id]
    ev = evaluate_placement(state, state.items[item_id], p)
    return ev.feasible, {"support": ev.support, "collision": ev.collision,
                         "boundary": ev.boundary, "load_violation": ev.load_violation,
                         "fragility_violation": ev.fragility_violation}


def verify_final_pack(state: PackingState, required=None):
    required = list(required or state.items)
    rows = {i: verify_placement(state, i) for i in required}
    return all(ok for ok, _ in rows.values()), rows
