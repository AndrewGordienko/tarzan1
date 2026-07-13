"""Analytical retention budgets for parallel-jaw grasps."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetentionBudget:
    mass_kg: float
    planned_acceleration_mps2: float
    gravity_mps2: float
    friction_coefficient: float
    safety_factor: float
    required_tangential_support_n: float
    target_total_normal_force_n: float
    target_normal_force_per_side_n: float
    maximum_allowed_normal_force_n: float
    available_support_at_limit_n: float
    tool_force_cap_n: float
    fragility_force_ceiling_n: float
    feasible: bool
    rejection_reason: str | None

    def as_dict(self):
        return asdict(self)


def calculate_retention_budget(
    mass_kg: float,
    planned_acceleration_mps2: float,
    friction_coefficient: float,
    *,
    safety_factor: float = 1.5,
    tool_force_cap_n: float = 125.0,
    fragility_force_ceiling_n: float = 125.0,
    gravity_mps2: float = 9.81,
) -> RetentionBudget:
    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if planned_acceleration_mps2 < 0:
        raise ValueError("planned_acceleration_mps2 cannot be negative")
    if friction_coefficient <= 0:
        raise ValueError("friction_coefficient must be positive")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least one")
    required = mass_kg * (gravity_mps2 + planned_acceleration_mps2)
    target_normal = safety_factor * required / friction_coefficient
    allowed = min(tool_force_cap_n, fragility_force_ceiling_n)
    available = friction_coefficient * allowed
    feasible = target_normal <= allowed
    reason = None if feasible else (
        "fragility_force_ceiling_exceeded"
        if fragility_force_ceiling_n < tool_force_cap_n
        else "tool_force_cap_exceeded"
    )
    return RetentionBudget(
        mass_kg=mass_kg,
        planned_acceleration_mps2=planned_acceleration_mps2,
        gravity_mps2=gravity_mps2,
        friction_coefficient=friction_coefficient,
        safety_factor=safety_factor,
        required_tangential_support_n=required,
        target_total_normal_force_n=target_normal,
        target_normal_force_per_side_n=target_normal / 2.0,
        maximum_allowed_normal_force_n=allowed,
        available_support_at_limit_n=available,
        tool_force_cap_n=tool_force_cap_n,
        fragility_force_ceiling_n=fragility_force_ceiling_n,
        feasible=feasible,
        rejection_reason=reason,
    )
