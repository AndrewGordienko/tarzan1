"""Coupled arm/tool gates for the product PackCell embodiment."""
from __future__ import annotations
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]

def evaluate_pair(arm: dict, tool: dict, requirements: dict) -> dict:
    reasons = []
    aperture = tool.get("usable_external_aperture_m")
    capacity = tool.get("workpiece_capacity_kg")
    tool_mass = tool.get("tool_mass_kg")
    combined = None if tool_mass is None else tool_mass + requirements["workpiece_mass_kg"]
    derated_required = None if combined is None else combined * requirements["arm_payload_safety_factor"]
    aperture_pass = aperture is not None and aperture >= requirements["minimum_usable_external_aperture_m"]
    tool_payload_pass = capacity is not None and capacity >= requirements["workpiece_mass_kg"]
    arm_payload_pass = derated_required is not None and arm["rated_payload_kg"] >= derated_required
    if not aperture_pass: reasons.append("usable_external_aperture_unverified_or_below_106mm")
    if not tool_payload_pass: reasons.append("tool_workpiece_capacity_below_8kg")
    if not arm_payload_pass: reasons.append("arm_payload_after_tool_and_safety_margin_insufficient_or_unknown")
    wrist_moment_nm = requirements["workpiece_mass_kg"] * 9.81 * requirements["worst_case_workpiece_cog_offset_m"]
    reasons.extend(["wrist_payload_curve_and_tool_cog_moment_not_yet_verified", "acceleration_derating_not_yet_verified", "fragility_force_range_not_yet_verified", "tool_inside_carton_collision_envelope_not_yet_verified"])
    sim_complete = bool(arm.get("simulation_asset") and tool.get("simulation_asset"))
    if not sim_complete: reasons.append("complete_provenance_pinned_pair_asset_missing")
    hard_pass = aperture_pass and tool_payload_pass and arm_payload_pass
    return {
        "pair_id": f'{arm["id"]}__{tool["id"]}',
        "role": arm["role"],
        "gates": {"aperture": aperture_pass, "tool_workpiece_capacity": tool_payload_pass, "arm_payload_derated": arm_payload_pass, "wrist_moment_verified": False, "acceleration_derating_verified": False, "fragility_force_verified": False, "tool_carton_clearance_verified": False, "full_simulation_asset": sim_complete},
        "combined_tool_and_workpiece_mass_kg": combined,
        "arm_payload_required_with_safety_margin_kg": derated_required,
        "workpiece_wrist_moment_proxy_nm": wrist_moment_nm,
        "tool_cog_moment_nm": None,
        "status": "candidate_pending_engineering_verification" if hard_pass else "rejected_or_incomplete",
        "reasons": reasons
    }

def build_matrix(path: str | Path | None = None) -> dict:
    source = Path(path or ROOT / "configs/coupled_embodiment_candidates_v1.json")
    data = json.loads(source.read_text())
    pairs = [evaluate_pair(arm, tool, data["requirements"]) for arm in data["arms"] for tool in data["tools"]]
    return {
        "schema": "coupled_embodiment_selection_v1",
        "scope": data["scope"],
        "requirements": data["requirements"],
        "pairs": pairs,
        "product_shortlist": [p["pair_id"] for p in pairs if p["role"] == "product_candidate" and p["status"] == "candidate_pending_engineering_verification"],
        "selected_pair": None,
        "decision": "no_pair_frozen; verify custom-finger aperture and wrist payload curves",
        "simulation_proxy_note": "UR10e/iiwa14 are geometry proxies only and must not be presented as the selected physical product."
    }
