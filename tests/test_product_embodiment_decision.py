from osc.packcell.product_embodiment_decision import ALLOWED_DECISIONS, build_decision

def test_frozen_aperture_is_not_relaxed():
    result = build_decision()
    assert result["frozen_requirement_m"] == .106
    assert result["requirement_relaxed"] is False

def test_confirmation_layouts_remain_sealed():
    result = build_decision()
    assert len(result["development_layouts"]) == 50
    assert result["confirmation_layouts_touched"] is False

def test_decision_is_bounded_and_no_pair_is_promoted_early():
    result = build_decision()
    assert result["decision"] in ALLOWED_DECISIONS
    assert result["decision"] == "no eligible pair"
    assert all(not tool["eligible"] for tool in result["tool_paths"])

def test_ur15_only_leads_on_proxy_not_final_eligibility():
    result = build_decision()
    ur15 = next(x for x in result["arm_envelopes"] if x["arm"] == "UR15")
    assert ur15["reach_proxy_pass"] and ur15["payload_pass"]
    assert not ur15["joint_margin_verified"]
