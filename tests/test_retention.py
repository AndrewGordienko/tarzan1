import pytest

from osc.packcell.retention import calculate_retention_budget


def test_retention_budget_uses_gravity_acceleration_friction_and_safety_factor():
    budget=calculate_retention_budget(2.,1.,.8,safety_factor=1.5)
    assert budget.required_tangential_support_n==pytest.approx(21.62)
    assert budget.target_total_normal_force_n==pytest.approx(40.5375)
    assert budget.available_support_at_limit_n==pytest.approx(100.)
    assert budget.feasible


def test_retention_budget_rejects_tool_or_fragility_limit():
    tool=calculate_retention_budget(8.,4.,.3,tool_force_cap_n=125.)
    assert not tool.feasible and tool.rejection_reason=='tool_force_cap_exceeded'
    fragile=calculate_retention_budget(2.,1.,.8,fragility_force_ceiling_n=20.)
    assert not fragile.feasible and fragile.rejection_reason=='fragility_force_ceiling_exceeded'


@pytest.mark.parametrize('field,value',[
    ('mass_kg',0.),('planned_acceleration_mps2',-1.),('friction_coefficient',0.)
])
def test_retention_budget_rejects_invalid_inputs(field,value):
    args={'mass_kg':1.,'planned_acceleration_mps2':0.,'friction_coefficient':.8}
    args[field]=value
    with pytest.raises(ValueError):calculate_retention_budget(**args)
