"""Deterministic geometric/quasistatic packing world model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .domain import PackItem, PackingState, Placement


@dataclass
class PlacementEvaluation:
    feasible: bool
    collision: bool = False
    boundary: bool = False
    support: float = 1.0
    load_violation: bool = False
    fragility_violation: bool = False
    risk: float = 0.0
    free_volume_after: float = 0.0


def overlap_xy(a: Placement, b: Placement) -> float:
    dx = max(0.0, min(a.max_corner[0], b.max_corner[0]) - max(a.position[0], b.position[0]))
    dy = max(0.0, min(a.max_corner[1], b.max_corner[1]) - max(a.position[1], b.position[1]))
    return dx * dy


def intersects(a: Placement, b: Placement) -> bool:
    return (overlap_xy(a, b) > 1e-8 and
            min(a.max_corner[2], b.max_corner[2]) > max(a.position[2], b.position[2]) + 1e-8)


def support_fraction(state: PackingState, placement: Placement) -> tuple[float, str | None]:
    base_area = placement.size[0] * placement.size[1]
    if placement.position[2] <= 1e-8:
        return 1.0, None
    best, support = 0.0, None
    for p in state.placements.values():
        if abs(p.max_corner[2] - placement.position[2]) > 1e-6:
            continue
        area = overlap_xy(p, placement)
        if area > best:
            best, support = area, p.item_id
    return (best / max(base_area, 1e-9), support)


def evaluate_placement(state: PackingState, item: PackItem, placement: Placement,
                       min_support: float = 0.85) -> PlacementEvaluation:
    cx, cy, cz = state.container.dimensions
    boundary = any(p < -1e-8 or q > lim + 1e-8
                   for p, q, lim in zip(placement.position, placement.max_corner,
                                        state.container.dimensions))
    collision = any(intersects(placement, other) for other in state.occupancy(item.item_id))
    support, support_id = support_fraction(state, placement)
    placement.support_id = support_id
    supported = support >= min_support
    load_above = sum(state.items[p.item_id].mass for p in state.occupancy(item.item_id)
                     if p.position[2] >= placement.max_corner[2] - 1e-8 and overlap_xy(p, placement) > 1e-8)
    load_violation = load_above > item.load_limit + 1e-8
    fragility = any(
        (item.fragile and p.position[2] >= placement.max_corner[2] - 1e-8
         and overlap_xy(p, placement) > 1e-8)
        or (state.items[p.item_id].fragile
            and placement.position[2] >= p.max_corner[2] - 1e-8
            and overlap_xy(p, placement) > 1e-8
            and item.mass > state.items[p.item_id].load_limit)
        for p in state.occupancy(item.item_id))
    total_load = state.packed_mass + item.mass
    load_violation = load_violation or total_load > state.container.max_load + 1e-8
    feasible = not (boundary or collision or not supported or load_violation or fragility)
    risk = float(boundary + collision + (not supported) + load_violation + fragility)
    free = state.container.volume - state.packed_volume - placement.volume
    return PlacementEvaluation(feasible, collision, boundary, support, load_violation,
                                fragility, risk, free)


def apply_placement(state: PackingState, placement: Placement) -> PackingState:
    out = state.clone()
    out.placements[placement.item_id] = placement
    out.staged.pop(placement.item_id, None)
    out.items[placement.item_id].already_packed = True
    out.history.append({"action": "place_inside", "item": placement.item_id,
                        "position": placement.position, "size": placement.size})
    return out


def remove_placement(state: PackingState, item_id: str, stage=True) -> PackingState:
    out = state.clone()
    old = out.placements.pop(item_id, None)
    if old is not None and stage:
        old.staged = True
        out.staged[item_id] = old
        out.history.append({"action": "temporarily_remove", "item": item_id})
    return out
