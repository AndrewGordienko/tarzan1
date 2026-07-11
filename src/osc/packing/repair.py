"""Belief-side counterfactual repair search for late packing arrivals."""
from __future__ import annotations

from dataclasses import dataclass, field

from .candidates import candidate_placements
from .domain import PackItem, PackingState, Placement
from .world_model import apply_placement, remove_placement


@dataclass
class RepairPlan:
    actions: list[dict] = field(default_factory=list)
    removed_item: str | None = None
    late_placement: Placement | None = None
    repack_placement: Placement | None = None
    score: tuple = ()
    certificate: dict = field(default_factory=dict)


def _staging_ok(item: PackItem, staging_region: tuple[float, float, float, float]) -> bool:
    sx, sy, sz, _ = staging_region
    return item.dimensions[0] <= sx and item.dimensions[1] <= sy and item.dimensions[2] <= sz


def search_counterfactual_repair(state: PackingState, late_item: PackItem,
                                 staging_region=(.9, .9, .9, 1.0)) -> RepairPlan | None:
    """Search direct placements, then one-object remove/place/repack plans.

    This function accepts only the camera-derived ``PackingState``. Scorer state,
    blocker labels, and fixture metadata are deliberately not parameters.
    """
    direct = list(candidate_placements(state, late_item))
    if direct:
        p = direct[0]
        return RepairPlan([{"kind": "place", "item": late_item.item_id, "placement": p.position}],
                          score=(0, 0, -p.volume), certificate={"direct": True, "search_complete": True})
    plans: list[RepairPlan] = []
    for item_id in sorted(state.placements):
        item = state.items[item_id]
        if not _staging_ok(item, staging_region):
            continue
        removed = remove_placement(state, item_id)
        for lp in candidate_placements(removed, late_item):
            after_late = apply_placement(removed, lp)
            for rp in candidate_placements(after_late, item):
                plans.append(RepairPlan(
                    [{"kind": "temporarily_remove", "item": item_id},
                     {"kind": "place", "item": late_item.item_id, "placement": lp.position},
                     {"kind": "repack", "item": item_id, "placement": rp.position}],
                    item_id, lp, rp, (1, 3, lp.position, rp.position),
                    {"direct": False, "search_complete": True, "candidate_count": len(plans) + 1}))
    return min(plans, key=lambda p: p.score) if plans else None
