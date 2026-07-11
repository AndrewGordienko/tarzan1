from osc.packing.domain import Container, PackItem, PackingState, Placement
from osc.packing.repair import search_counterfactual_repair


def _state(blocker_size=(.8, .8, .2)):
    c = Container("box", (1.0, 1.0, 1.0))
    blocker = PackItem("blocker", blocker_size, fragile=True, load_limit=.1)
    late = PackItem("late", (.8, .8, .6))
    return PackingState(c, {"blocker": blocker, "late": late},
                        {"blocker": Placement("blocker", (0, 0, 0), blocker_size)})


def test_repair_search_finds_remove_place_repack():
    s = _state(); plan = search_counterfactual_repair(s, s.items["late"])
    assert plan and plan.removed_item == "blocker"
    assert [a["kind"] for a in plan.actions] == ["temporarily_remove", "place", "repack"]


def test_direct_control_has_zero_removals():
    s = _state((.2, .2, .2))
    s.placements["blocker"] = Placement("blocker", (0, 0, 0), (.2, .2, .2))
    s.items["late"] = PackItem("late", (.2, .2, .2))
    plan = search_counterfactual_repair(s, s.items["late"])
    assert plan and plan.removed_item is None and len(plan.actions) == 1


def test_staging_rejection_and_partial_plan_rejection():
    s = _state(); assert search_counterfactual_repair(s, s.items["late"], staging_region=(.1, .1, .1, 1.0)) is None
