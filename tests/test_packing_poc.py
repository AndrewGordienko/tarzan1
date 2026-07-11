from osc.packing.benchmark import run_episode, scenarios
from osc.packing.compiler import compile_packing_demo, compile_with_inferred_posterior, POLICY_CATALOG
from osc.packing.benchmark import run_demo_dependence
from osc.packing.domain import PackingState
from osc.packing.planner import PackingPlanner


def test_demo_compiles_persistent_constraints_and_actions():
    p = compile_packing_demo()
    assert "every order item must be inside the shipping box" in p.objective
    assert {c.kind for c in p.hard_constraints} >= {"containment", "collision", "support", "load"}
    assert "temporarily_remove" in p.available_actions


def test_oracle_packing_handles_changed_inventory_and_impossible_order():
    ss = scenarios()
    good = [run_episode(s, "oracle", 0) for s in ss[:3]]
    bad = run_episode(ss[3], "oracle", 0)
    assert all(r["success"] for r in good)
    assert bad["success"] is False


def test_late_large_item_causes_genuine_rearrangement():
    r = run_episode(scenarios()[2], "oracle", 0)
    assert r["success"]
    assert r["rearranged"]
    assert any(x["kind"] == "temporarily_remove" for x in r["actions"])


def test_demo_causally_changes_program_posterior_and_arrangement():
    rows = run_demo_dependence()
    assert rows["correct_heavy_bottom"]["program_policy"] == "heavy_bottom_fragile_top"
    assert rows["different_max_volume"]["program_policy"] == "maximize_volume"
    assert rows["minimize_rehandling"]["program_policy"] == "minimize_rehandling"
    assert rows["correct_heavy_bottom"]["arrangement"] != rows["different_max_volume"]["arrangement"]
    assert rows["correct_heavy_bottom"]["policy_behavior_match"]
    assert rows["different_max_volume"]["policy_behavior_match"]
    assert not rows["no_demo"]["policy_behavior_match"]
    assert rows["shuffled_demo"]["program_policy"] != rows["correct_heavy_bottom"]["program_policy"]
    assert rows["oracle_task_program"]["program_policy"] == "heavy_bottom_fragile_top"


def test_inverse_planning_keeps_ground_truth_program_in_candidate_set():
    p = compile_with_inferred_posterior()
    assert set(p.posterior) == set(POLICY_CATALOG)
    assert abs(sum(p.posterior.values()) - 1.0) < 1e-9
