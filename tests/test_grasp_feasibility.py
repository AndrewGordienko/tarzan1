from pathlib import Path

from osc.packcell.grasp_feasibility import compatibility_matrix, load_end_effector_contract, evaluate_dimensions


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


def test_confirmation_objects_are_not_read_by_matrix():
    result = compatibility_matrix(ROOT / "configs/amazon_small_sortable_v1.json", load_end_effector_contract())
    assert result["confirmation_objects_sealed"] is True
    assert all(row["sealed_set"] == "development" for row in result["objects"])
