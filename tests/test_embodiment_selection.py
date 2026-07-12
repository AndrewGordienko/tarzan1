from osc.packcell.embodiment_selection import build_matrix

def test_robotiq_2f140_is_rejected_for_tool_payload():
    matrix = build_matrix()
    pairs = [p for p in matrix["pairs"] if p["pair_id"].endswith("robotiq_2f140")]
    assert pairs and all(not p["gates"]["tool_workpiece_capacity"] for p in pairs)

def test_default_2fg14_requires_aperture_verification():
    matrix = build_matrix()
    pairs = [p for p in matrix["pairs"] if p["pair_id"].endswith("onrobot_2fg14_default_external")]
    assert pairs and all(not p["gates"]["aperture"] for p in pairs)

def test_no_product_pair_is_frozen_without_wrist_and_asset_verification():
    matrix = build_matrix()
    assert matrix["selected_pair"] is None
    assert all(not p["gates"]["wrist_moment_verified"] for p in matrix["pairs"])
