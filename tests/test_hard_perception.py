"""Contract tests for the stratified re-audit and real viewpoint loop."""
from types import SimpleNamespace

from osc.benchmark.hard_perception import (hard_perception_splits, paired_configs,
                                           validate_identifiability_metric)
from osc.benchmark.runner import run_benchmark
from osc.execution.loop import ExecConfig


def test_hard_split_has_the_three_required_categories_and_camera_geometry():
    splits = hard_perception_splits()
    assert [s.name for s in splits] == ["hard_noisy_visible", "hard_hidden_default_view",
                                        "hard_fundamental_ambiguity"]
    assert all(s.rand.camera_model for s in splits)
    assert all(s.corr.size_noise > 0 and s.corr.occlusion_prob > 0 and s.corr.drop_prob > 0
               and s.corr.identity_swap_prob > 0 for s in splits)
    configs = paired_configs()
    assert configs["oracle_perception"][1] and not configs["oracle_perception"][2]
    assert configs["oracle_role_binding"][2]


def test_hidden_feature_category_executes_a_real_viewpoint_move():
    cfg = ExecConfig(resolution=True, allow_inspection=True, allow_viewpoint=True,
                     allow_clarification=True, commit_threshold=0.99)
    # Seed 0 deliberately contains enough ambiguity to exhaust passive evidence.
    recs = run_benchmark(splits=[hard_perception_splits()[1]], seeds=[0], cfg=cfg)
    assert any(r.viewpoints for r in recs)
    assert all(r.viewpoint_actions >= len(r.viewpoints) for r in recs)
    assert any(r.viewpoint_diagnostics for r in recs)


def test_identifiability_metric_controls_validate_direction_and_strata():
    recs = [SimpleNamespace(identifiable=True, role_confidence=.9, split="a"),
            SimpleNamespace(identifiable=False, role_confidence=.1, split="a"),
            SimpleNamespace(identifiable=True, role_confidence=.8, split="b"),
            SimpleNamespace(identifiable=False, role_confidence=.2, split="b")]
    d = validate_identifiability_metric(recs)
    assert d["ground_truth_identifiability_auroc"] == 1.0
    assert d["pooled_auroc"] == 1.0
    assert d["negated_score_auroc"] == 0.0
    assert d["label_balance"] == {"identifiable": 2, "ambiguous": 2, "n": 4}


def test_identifiability_labels_are_fixed_at_initial_scene_across_configs():
    splits = hard_perception_splits()
    low = ExecConfig(resolution=True, allow_inspection=False, allow_clarification=True)
    high = ExecConfig(resolution=True, allow_inspection=True, allow_viewpoint=True,
                      allow_clarification=True, commit_threshold=.99)
    a = run_benchmark(splits=splits, seeds=[0], cfg=low)
    b = run_benchmark(splits=splits, seeds=[0], cfg=high)
    labels = lambda rs: {(r.task, r.split, r.seed): r.identifiable for r in rs}
    assert labels(a) == labels(b)
