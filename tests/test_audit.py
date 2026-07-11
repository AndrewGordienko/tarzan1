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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
