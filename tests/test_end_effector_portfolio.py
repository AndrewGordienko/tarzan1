from osc.packcell.end_effector_portfolio import ObjectToolBelief, select_tool

def test_small_porous_object_uses_jaws():
    result = select_tool(ObjectToolBelief(1., .05, 0., .1, True, False, 30.))
    assert result["decision"] == "use_jaws"

def test_large_sealed_object_uses_suction():
    result = select_tool(ObjectToolBelief(4., .15, .01, .97, False, False, 50.))
    assert result["decision"] == "use_suction"

def test_unknown_surface_abstains():
    result = select_tool(ObjectToolBelief(2., .10, .01, .5, None, False, 30.))
    assert result["decision"] == "abstain"
