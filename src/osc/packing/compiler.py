"""Compile one packing demonstration into persistent constraints/preferences."""
from __future__ import annotations

from .demonstration import canonical_demo_events, parse_demo
from .domain import PackingConstraint, PackingProgram


def compile_packing_demo(events=None) -> PackingProgram:
    events = parse_demo(events or canonical_demo_events())
    kinds = [e.kind for e in events]
    hard = [
        PackingConstraint("inside_container", "containment"),
        PackingConstraint("no_intersections", "collision"),
        PackingConstraint("supported", "support"),
        PackingConstraint("respect_load_limits", "load"),
        PackingConstraint("fragile_not_support_heavy", "fragility"),
    ]
    prefs = [
        PackingConstraint("heavy_low", "ordering", hard=False, weight=1.5),
        PackingConstraint("fragile_high", "ordering", hard=False, weight=1.0),
        PackingConstraint("preserve_large_item_space", "future_fit", hard=False, weight=2.0),
        PackingConstraint("minimize_rehandling", "rehandling", hard=False, weight=1.0),
        PackingConstraint("minimize_unused_volume", "volume", hard=False, weight=.5),
    ]
    return PackingProgram(
        objective=["every order item must be inside the shipping box"],
        hard_constraints=hard, preferences=prefs,
        available_actions=["inspect", "pick", "place", "temporarily_remove",
                           "rearrange", "verify", "request_metadata", "clarify"],
        ordering_rules=["heavy_before_fragile", "large_before_small_when_fit_is_tight"],
        demonstration_events=[e.to_dict() for e in events],
        hidden_information=["mass", "fragility", "load_limit"],
    )
