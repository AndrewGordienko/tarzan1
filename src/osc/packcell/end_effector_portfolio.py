"""Tool selection across a portfolio, with explicit abstention."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ObjectToolBelief:
    mass_kg: float
    width_m: float
    flat_seal_area_m2: float
    suction_confidence: float
    porous: bool | None
    fragile: bool | None
    jaw_force_required_n: float | None

def select_tool(obj: ObjectToolBelief, *, jaw_aperture_m: float = .106, jaw_payload_kg: float = 8., suction_payload_kg: float = 8., minimum_suction_confidence: float = .90) -> dict:
    jaw_ok = obj.mass_kg <= jaw_payload_kg and obj.width_m <= jaw_aperture_m and obj.jaw_force_required_n is not None and obj.jaw_force_required_n <= 80.
    suction_ok = obj.mass_kg <= suction_payload_kg and obj.porous is False and obj.flat_seal_area_m2 > .0025 and obj.suction_confidence >= minimum_suction_confidence
    if obj.fragile is None or obj.porous is None:
        return {"decision": "abstain", "reason": "unknown_surface_or_fragility", "eligible_tools": []}
    eligible = [name for name, ok in (("jaws", jaw_ok), ("suction", suction_ok)) if ok]
    if suction_ok and obj.width_m > jaw_aperture_m: decision = "use_suction"
    elif jaw_ok: decision = "use_jaws"
    elif suction_ok: decision = "use_suction"
    else: decision = "abstain"
    return {"decision": decision, "reason": None if eligible else "no_installed_tool_covers_object", "eligible_tools": eligible}
