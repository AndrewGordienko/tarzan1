from osc.packing.benchmark import run_episode, scenarios
from osc.packing.compiler import compile_packing_demo
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
