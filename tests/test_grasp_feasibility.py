from pathlib import Path

from osc.packcell.grasp_feasibility import compatibility_matrix, load_end_effector_contract, evaluate_dimensions, select_grasp_axis
from osc.packcell.grasp_pose_generator import generate_grasp_pose


ROOT = Path(__file__).resolve().parents[1]


def test_stock_panda_contract_reads_measured_usable_aperture():
    c = load_end_effector_contract()
    assert c.max_usable_aperture_m < 0.085
    assert c.insertion_clearance_m > 0
    assert c.safe_payload_kg == 1.5


def test_matrix_rotates_medium_object_but_rejects_large_object():
    c = load_end_effector_contract()
    medium = evaluate_dimensions((0.12, 0.08, 0.06), c)
    large = evaluate_dimensions((0.25, 0.15, 0.10), c)
    assert medium["feasible_axes"] == [2]
    assert large["feasible"] is False
    assert select_grasp_axis((0.12, 0.08, 0.06), c) == 2
    assert select_grasp_axis((0.25, 0.15, 0.10), c) is None


def test_confirmation_objects_are_not_read_by_matrix():
    result = compatibility_matrix(ROOT / "configs/amazon_small_sortable_v1.json", load_end_effector_contract())
    assert result["confirmation_objects_sealed"] is True
    assert all(row["sealed_set"] == "development" for row in result["objects"])

def test_geometry_conditioned_pose_generator_selects_maximin_or_abstains():
    candidates = [{"offset_grasp_frame_m": [0, 0, .002], "structural_clearance_m": .004, "left_pad_clearance_m": .004, "right_pad_clearance_m": .004, "pad_overlap_m": .01, "robust_clearance_m": .003}, {"offset_grasp_frame_m": [0, 0, .006], "structural_clearance_m": .006, "left_pad_clearance_m": .006, "right_pad_clearance_m": .006, "pad_overlap_m": .01, "robust_clearance_m": .005}]
    assert generate_grasp_pose(candidates)["pose"]["offset_grasp_frame_m"] == [0.0, 0.0, .006]
    candidates[1]["robust_clearance_m"] = .001
    assert generate_grasp_pose(candidates)["status"] == "selected"
    assert generate_grasp_pose([{**candidates[0], "robust_clearance_m": .001}])["status"] == "abstain"
