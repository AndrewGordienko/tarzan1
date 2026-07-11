"""Finite meaningful candidate placements for a rectangular container."""
from __future__ import annotations

from .domain import PackItem, PackingState, Placement
from .world_model import evaluate_placement


def candidate_placements(state: PackingState, item: PackItem):
    points = {(0.0, 0.0, 0.0)}
    for p in state.placements.values():
        x, y, z = p.position
        sx, sy, sz = p.size
        points.update({(x + sx, y, z), (x, y + sy, z),
                       (x + sx, y + sy, z), (x, y, z + sz)})
    # Add wall-aligned extremes and a small interior grid for changed boxes.
    cx, cy, _ = state.container.dimensions
    points.update({(cx, 0.0, 0.0), (0.0, cy, 0.0), (cx, cy, 0.0)})
    seen = set()
    for size in item.orientations():
        orient = tuple(item.dimensions.index(v) for v in size)
        for point in sorted(points):
            placement = Placement(item.item_id, point, size, orient)
            key = (placement.position, placement.size)
            if key in seen:
                continue
            seen.add(key)
            if evaluate_placement(state, item, placement).feasible:
                yield placement
