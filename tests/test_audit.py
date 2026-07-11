"""Discrimination-audit helper tests + the decisive finding is stable."""
from osc.benchmark import discrimination_audit as A


def test_auroc_perfect_and_random():
    # perfectly separating score -> AUROC 1.0; constant score -> 0.5
    assert A._auroc([0.1, 0.2, 0.9, 0.95], [False, False, True, True]) == 1.0
    assert A._auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_best_coverage_at_risk_monotone():
    scores = [0.9, 0.8, 0.7, 0.2]
    correct = [True, True, False, False]
    # at risk<=0 we can only admit the two confident-correct -> coverage 0.5
    assert A._best_coverage_at_risk(scores, correct, 0.0) == 0.5


def test_operating_point_returns_low_coverage_when_inseparable():
    # identical score distributions for both classes => no threshold separates them
    scores = [0.6] * 10
    ident = [True, False] * 5
    cov, tau, amb = A._operating_point(scores, ident, amb_budget=0.05)
    assert cov == 0.0            # cannot commit on identifiable without also committing on ambiguous


def test_audit_populates_features_and_labels():
    d = A.audit(seeds=range(10))
    assert set(d["features"]) >= {"confidence(min-marginal)", "role_margin(top1-top2)",
                                  "assignment_margin", "neg_entropy"}
    assert len(d["conf"]) == len(d["bind"]) == len(d["ident"]) > 0


def test_multivariate_probe_beats_single_feature_on_identifiability():
    mp = A.multivariate_probe(dev_seeds=range(0, 30), held_seeds=range(5000, 5030))
    # combining weak features must rank identifiability well above chance and above
    # the ~0.75 single-feature ceiling (validates that combos help; route not closed).
    auroc = mp["models"]["identifiable"]["logistic"]["auroc"]
    assert auroc > 0.80
    # and binding-correctness stays comparatively weak (needs more evidence, not model)
    assert mp["models"]["binding"]["logistic"]["auroc"] < auroc


def test_clarification_decomposition_separates_binding_from_control():
    d = A.clarification_decomposition(seeds=range(40))
    assert d["n"] > 0
    # control is strong given a correct binding; the loss is binding persistence
    assert d["success_given_binding"] > 0.9
    assert d["persistent_binding"] < d["success_given_binding"]
    bp = d["breakpoints"]
    assert bp["binding_not_persistent"] > bp["correct_binding_control_fail"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
