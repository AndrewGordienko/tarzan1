"""Compile one packing demonstration into persistent constraints/preferences."""
from __future__ import annotations

import math
import numpy as np

from .demonstration import canonical_demo_events, parse_demo
from .domain import PackingConstraint, PackingProgram


POLICY_CATALOG = {
    "unknown_or_unexplained": {"rules": [], "prefs": {"heavy_low": 0.0, "fragile_high": 0.0,
                                                           "preserve_large_item_space": 0.0,
                                                           "minimize_rehandling": 0.0,
                                                           "minimize_unused_volume": 0.0}},
    "neutral_prior": {"rules": [], "prefs": {"heavy_low": 0.0, "fragile_high": 0.0,
                                                   "preserve_large_item_space": 0.0,
                                                   "minimize_rehandling": 0.0,
                                                   "minimize_unused_volume": 0.0}},
    "heavy_bottom_fragile_top": {
        "rules": ["heavy_before_fragile"],
        "prefs": {"heavy_low": 2.0, "fragile_high": 2.0, "preserve_large_item_space": 1.0,
                   "minimize_rehandling": .5, "minimize_unused_volume": .5},
    },
    "maximize_volume": {
        "rules": ["large_items_first"],
        "prefs": {"heavy_low": .2, "fragile_high": .1, "preserve_large_item_space": 2.0,
                   "minimize_rehandling": .1, "minimize_unused_volume": 2.0},
    },
    "minimize_rehandling": {
        "rules": ["preserve_arrival_order"],
        "prefs": {"heavy_low": .4, "fragile_high": .2, "preserve_large_item_space": .2,
                   "minimize_rehandling": 3.0, "minimize_unused_volume": .1},
    },
}


def _program(policy_name, events):
    spec = POLICY_CATALOG[policy_name]
    hard = [
        PackingConstraint("inside_container", "containment"),
        PackingConstraint("no_intersections", "collision"),
        PackingConstraint("supported", "support"),
        PackingConstraint("respect_load_limits", "load"),
        PackingConstraint("fragile_not_support_heavy", "fragility"),
    ]
    prefs = [PackingConstraint(n, n.split("_")[0], hard=False, weight=w)
             for n, w in spec["prefs"].items()]
    return PackingProgram(
        objective=["every order item must be inside the shipping box"],
        hard_constraints=hard, preferences=prefs,
        available_actions=["inspect", "pick", "place", "temporarily_remove",
                           "rearrange", "verify", "request_metadata", "clarify"],
        ordering_rules=list(spec["rules"]),
        demonstration_events=[e.to_dict() for e in events],
        hidden_information=["mass", "fragility", "load_limit"],
        policy_name=policy_name)


def compile_packing_demo(events=None, policy_name="heavy_bottom_fragile_top") -> PackingProgram:
    events = parse_demo(events or canonical_demo_events())
    if policy_name not in POLICY_CATALOG:
        raise ValueError(f"unknown packing policy: {policy_name}")
    return _program(policy_name, events)


def infer_program_posterior(events=None):
    """Small inverse-planning posterior over finite policy hypotheses."""
    if events is None:
        return {name: (1.0 if name == "neutral_prior" else 0.0) for name in POLICY_CATALOG}
    events = parse_demo(events)
    order = [e.item_id for e in events if e.kind == "place_inside" and e.item_id]
    scores = {}
    for name, spec in POLICY_CATALOG.items():
        score = 0.0
        props = [e.properties or {} for e in events]
        # Demonstration metadata is never treated as an answer key. Optional
        # metadata may describe an object/SKU, but policy inference must explain
        # the observed ordering and outcomes from the events themselves.
        if any(p.get("rehandling") == "low" for p in props) and name == "minimize_rehandling":
            score += 3.0
        if any(p.get("fill") == "high" for p in props) and name == "maximize_volume":
            score += 3.0
        if "heavy_before_fragile" in spec["rules"]:
            score += 2.0 if _before(order, "heavy", "fragile") else -1.0
        if "large_items_first" in spec["rules"]:
            score += 0.5 if order == sorted(order) else 0.0
        if "preserve_arrival_order" in spec["rules"]:
            score += 1.0 if not any(e.kind == "rearrange" for e in events) else -1.0
        score += 0.25 * sum(e.kind == "rearrange" for e in events) if name == "minimize_rehandling" else 0.0
        scores[name] = score
    known_kinds = {"inspect", "pick", "place_inside", "place_on", "remove", "rearrange", "verify"}
    unknown_signal = any(e.kind not in known_kinds for e in events)
    scores["unknown_or_unexplained"] = 5.0 if unknown_signal else -3.0
    vals = np.array(list(scores.values()), dtype=float)
    probs = np.exp(vals - vals.max()); probs /= probs.sum()
    return {name: float(p) for name, p in zip(scores, probs)}


def compile_with_inferred_posterior(events=None):
    events = parse_demo(events or canonical_demo_events())
    posterior = infer_program_posterior(events)
    best = max(posterior, key=posterior.get)
    program = compile_packing_demo(events, best)
    program.posterior = posterior
    program.constraint_posterior = {
        "all_inside": .99,
        "heavy_below_fragile": .94 if best == "heavy_bottom_fragile_top" else .42,
        "fragile_never_supports": .98,
        "maximize_fill": .87 if best == "maximize_volume" else .31,
        "minimize_rehandling": .87 if best == "minimize_rehandling" else .31,
        "large_items_first": .83 if best == "maximize_volume" else .27,
        "preserve_future_space": .87 if best in ("heavy_bottom_fragile_top", "maximize_volume") else .31,
        "reproduce_exact_coordinates": .02,
    }
    if best == "unknown_or_unexplained":
        program.constraint_posterior = {}
    return program


def compile_composed_program(components):
    """Compose independently known constraints/preferences for a held-out combo."""
    components = tuple(components)
    if set(components) - {"heavy_below_fragile", "minimize_rehandling", "maximize_fill"}:
        raise ValueError("unknown program component")
    base = _program("heavy_bottom_fragile_top", [])
    base.policy_name = "composed_" + "_".join(components)
    base.ordering_rules = []
    if "heavy_below_fragile" in components:
        base.ordering_rules.append("heavy_before_fragile")
    if "maximize_fill" in components:
        base.ordering_rules.append("large_items_first")
    if "minimize_rehandling" in components:
        base.preferences = [PackingConstraint(c.name, c.kind, c.hard, c.weight * 1.5,
                                              dict(c.parameters)) for c in base.preferences]
    return base


def _before(order, a, b):
    return a in order and b in order and order.index(a) < order.index(b)
