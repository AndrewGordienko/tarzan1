"""Contract-level grasp feasibility, independent of controller execution."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class EndEffectorContract:
    name: str
    min_usable_aperture_m: float
    max_usable_aperture_m: float
    insertion_clearance_m: float
    pad_dimensions_m: tuple[float, float, float]
    closing_axis: str
    grip_force_range_n: tuple[float, float]
    payload_capacity_kg: float
    payload_safety_factor: float
    supported_grasp_axes: tuple[int, ...]
    fragile_materials: tuple[str, ...]

    @property
    def safe_payload_kg(self) -> float:
        return self.payload_capacity_kg / self.payload_safety_factor

def load_end_effector_contract(path: str | Path | None = None) -> EndEffectorContract:
    p = Path(path or Path(__file__).resolve().parents[3] / "configs" / "industrial_arm_panda_v1.json")
    x = json.loads(p.read_text()); g = x["gripper"]
    return EndEffectorContract(g["name"], float(g["min_usable_aperture_m"]), float(g["max_usable_aperture_m"]), float(g["insertion_clearance_m"]), tuple(g["pad_dimensions_m"]), g["closing_axis"], tuple(g["grip_force_range_n"]), float(g["payload_capacity_kg"]), float(g["payload_safety_factor"]), tuple(g["supported_grasp_axes"]), tuple(g["fragile_materials"]))

def evaluate_dimensions(dimensions_m: tuple[float, float, float], contract: EndEffectorContract) -> dict:
    rows = []
    for axis in contract.supported_grasp_axes:
        width = float(dimensions_m[axis]); required = width + 2 * contract.insertion_clearance_m
        rows.append({"closing_axis_index": axis, "closing_axis_width_m": width, "required_aperture_with_clearance_m": required, "feasible": contract.min_usable_aperture_m <= required <= contract.max_usable_aperture_m, "margin_to_max_aperture_m": contract.max_usable_aperture_m - required})
    feasible = [r for r in rows if r["feasible"]]
    return {"dimensions_m": list(dimensions_m), "axes": rows, "feasible": bool(feasible), "feasible_axes": [r["closing_axis_index"] for r in feasible]}

def select_grasp_axis(dimensions_m: tuple[float, float, float], contract: EndEffectorContract) -> int | None:
    """Choose the narrowest feasible closing axis, leaving maximum clearance."""
    result = evaluate_dimensions(dimensions_m, contract)
    feasible = [r for r in result["axes"] if r["feasible"]]
    return min(feasible, key=lambda r: r["required_aperture_with_clearance_m"])["closing_axis_index"] if feasible else None

def compatibility_matrix(task_path: str | Path, contract: EndEffectorContract) -> dict:
    task = json.loads(Path(task_path).read_text()); rows = []
    for i, dims in enumerate(task["objects"]["dimensions_m"]):
        result = evaluate_dimensions(tuple(dims), contract); result.update(object_index=i, sealed_set="development"); rows.append(result)
    return {"schema": "end_effector_grasp_compatibility_v1", "task_scope": task["scope"], "confirmation_objects_sealed": True, "contract": {"name": contract.name, "min_usable_aperture_m": contract.min_usable_aperture_m, "max_usable_aperture_m": contract.max_usable_aperture_m, "insertion_clearance_m": contract.insertion_clearance_m, "closing_axis": contract.closing_axis, "payload_capacity_kg": contract.payload_capacity_kg, "safe_payload_kg": contract.safe_payload_kg}, "objects": rows, "summary": {"total": len(rows), "feasible_objects": sum(r["feasible"] for r in rows), "infeasible_objects": sum(not r["feasible"] for r in rows)}}
