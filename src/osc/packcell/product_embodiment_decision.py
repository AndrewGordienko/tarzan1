"""Bounded UR15/UR20 and end-effector procurement decision."""
from __future__ import annotations
import math, random

ALLOWED_DECISIONS = {"UR15 + verified custom 2FG14", "UR15 + hybrid tool", "UR20 + verified custom 2FG14", "UR20 + hybrid tool", "no eligible pair"}

def development_waypoints(count: int = 50) -> list[dict]:
    rng = random.Random(8062026)
    layouts = []
    for i in range(count):
        pick = [rng.uniform(.25, .85), rng.uniform(-.30, .30), .80]
        layouts.append({"layout_id": f"DEV-{i:03d}", "pick": pick, "pregrasp": [pick[0], pick[1], .95], "lift": [pick[0], pick[1], 1.05], "box_corners": [[x, y, 1.00] for x in (.35, .75) for y in (-.60, -.30)], "deepest_placement": [.55, -.45, .78], "staging": [.55, .45, .82]})
    return layouts

def arm_envelope(arm: dict, layouts: list[dict], *, tool_length_m: float, tool_mass_kg: float) -> dict:
    base = [0., 0., .75]
    points = [(layout["layout_id"], name, point) for layout in layouts for name in ("pick", "pregrasp", "lift", "deepest_placement", "staging") for point in [layout[name]]]
    points += [(layout["layout_id"], "box_corner", point) for layout in layouts for point in layout["box_corners"]]
    rows = []
    for lid, phase, p in points:
        tcp_distance = math.dist(base, p); flange_proxy = tcp_distance + tool_length_m
        rows.append({"layout_id": lid, "phase": phase, "tcp_distance_m": tcp_distance, "conservative_flange_reach_proxy_m": flange_proxy, "reach_margin_m": arm["reach_m"] - flange_proxy})
    combined = tool_mass_kg + 8.; required = combined * 1.2
    return {"arm": arm["id"], "waypoint_count": len(rows), "maximum_reach_proxy_m": max(r["conservative_flange_reach_proxy_m"] for r in rows), "minimum_reach_margin_m": min(r["reach_margin_m"] for r in rows), "reach_proxy_pass": all(r["reach_margin_m"] >= .05 for r in rows), "combined_mass_kg": combined, "payload_with_20pct_margin_kg": required, "rated_payload_kg": arm["payload_kg"], "payload_pass": arm["payload_kg"] >= required, "joint_margin_verified": False, "collision_clearance_verified": False, "payload_curve_verified": False, "acceleration_derating_verified": False, "cycle_time_s": None, "cycle_time_reason": "manufacturer acceleration profile and executed joint paths unavailable", "footprint_diameter_m": arm["footprint_m"], "rows": rows}

def build_decision() -> dict:
    layouts = development_waypoints()
    arms = [{"id": "UR15", "payload_kg": 15., "reach_m": 1.30, "footprint_m": .204}, {"id": "UR20", "payload_kg": 20., "reach_m": 1.75, "footprint_m": .245}]
    # Default 2FG14 dimensions; custom-finger mass/CoG remain unknown.
    envelopes = [arm_envelope(a, layouts, tool_length_m=.1552, tool_mass_kg=1.5) for a in arms]
    object_moment = 8. * 9.81 * (.1552 + .125)
    tool_moment_proxy = 1.5 * 9.81 * (.1552 / 2)
    tool_paths = [
        {"id": "custom_2fg14", "base_external_opening_m": .105, "required_opening_m": .106, "preferred_procurement_opening_m": .110, "manufacturer_custom_fingers_supported": True, "custom_finger_dimensions_verified": False, "custom_tool_mass_cog_verified": False, "finger_platform_moment_limit_nm": {"x": 30., "y": 25.}, "estimated_static_moment_proxy_nm": object_moment + tool_moment_proxy, "small_object_closure_verified": False, "carton_entry_clearance_verified": False, "force_range_n": [40., 280.], "fragility_force_compatibility_verified": False, "eligible": False},
        {"id": "hybrid_jaw_suction", "jaw_opening_verified": False, "vacuum_surface_classes": ["nonporous_flat", "sealed_carton_pending_test"], "unknown_surface_policy": "jaws_or_abstain", "seal_quality_model_verified": False, "porosity_model_verified": False, "suction_payload_with_acceleration_verified": False, "tool_mass_cog_verified": False, "carton_entry_clearance_verified": False, "eligible": False}
    ]
    return {"schema": "product_embodiment_decision_v1", "frozen_requirement_m": .106, "requirement_relaxed": False, "development_layouts": layouts, "confirmation_layouts_touched": False, "arm_envelopes": envelopes, "tool_paths": tool_paths, "arm_preference_if_tool_verified": "UR15" if envelopes[0]["reach_proxy_pass"] and envelopes[0]["payload_pass"] else "UR20", "decision": "no eligible pair", "decision_reason": "UR15 clears the geometric reach/payload proxy, but exact joint/collision/payload-curve gates are unverified; neither custom 2FG14 nor hybrid tool has a verified physical contract."}
