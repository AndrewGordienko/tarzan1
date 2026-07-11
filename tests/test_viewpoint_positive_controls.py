from osc.benchmark.viewpoint_controls import run_viewpoint_positive_controls


def test_nonredundant_geometry_and_marker_viewpoint_controls():
    for control in run_viewpoint_positive_controls().values():
        top = control["oracle_tracks_top_only"]
        right = control["oracle_tracks_correct_view"]
        wrong = control["oracle_tracks_wrong_view"]
        assert top["ambiguous"] and not top["binding_correct"]
        assert right["binding_correct"] and not right["ambiguous"]
        assert wrong["ambiguous"] and not wrong["binding_correct"]
