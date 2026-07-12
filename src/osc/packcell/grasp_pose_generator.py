"""Object-conditioned, maximin grasp-pose selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

@dataclass(frozen=True)
class GraspPoseCandidate:
    offset_grasp_frame_m: tuple[float, float, float]
    structural_clearance_m: float
    left_pad_clearance_m: float
    right_pad_clearance_m: float
    pad_overlap_m: float
    palm_clearance_m: float
    fingertip_clearance_m: float
    robust_clearance_m: float
    active_contacts: int = 0

    @property
    def nominal_clearance_m(self) -> float:
        return min(self.structural_clearance_m, self.left_pad_clearance_m, self.right_pad_clearance_m, self.palm_clearance_m, self.fingertip_clearance_m)

def select_maximin(candidates: Iterable[GraspPoseCandidate], *, min_clearance_m: float = .002, min_pad_overlap_m: float = .005) -> GraspPoseCandidate | None:
    feasible = [c for c in candidates if c.active_contacts == 0 and c.nominal_clearance_m >= min_clearance_m and c.robust_clearance_m >= min_clearance_m and c.pad_overlap_m >= min_pad_overlap_m]
    return max(feasible, key=lambda c: (c.robust_clearance_m, c.nominal_clearance_m, -sum(abs(x) for x in c.offset_grasp_frame_m))) if feasible else None

def generate_grasp_pose(candidates: Iterable[Mapping[str, object]]) -> dict:
    parsed = [GraspPoseCandidate(tuple(float(x) for x in c["offset_grasp_frame_m"]), float(c["structural_clearance_m"]), float(c["left_pad_clearance_m"]), float(c["right_pad_clearance_m"]), float(c["pad_overlap_m"]), float(c.get("palm_clearance_m", c["structural_clearance_m"])), float(c.get("fingertip_clearance_m", c["structural_clearance_m"])), float(c["robust_clearance_m"]), int(c.get("active_contacts", 0))) for c in candidates]
    selected = select_maximin(parsed)
    return {"status": "selected" if selected else "abstain", "reason": None if selected else "no_robust_clearance_pose", "pose": None if selected is None else {"offset_grasp_frame_m": list(selected.offset_grasp_frame_m), "robust_clearance_m": selected.robust_clearance_m, "nominal_clearance_m": selected.nominal_clearance_m, "pad_overlap_m": selected.pad_overlap_m}}
