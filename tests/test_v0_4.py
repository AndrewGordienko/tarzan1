"""v0.4 resolution-layer tests: the policy escalates inspection -> clarification
-> abstain and never guesses; clarification is asked once and persists; the
benchmark exposes the autonomy-vs-safety trade-off.
"""
from dataclasses import replace

from osc.execution.loop import ExecConfig
from osc.execution.resolution import (ResolutionConfig, ResolutionPolicy,
                                       TaskContext)
from osc.skills.correspondence import RoleAssignment
from osc.benchmark.runner import run_benchmark
from osc.metrics.metrics import aggregate


def _ra(per_role_conf):
    return RoleAssignment(mapping={r: r for r in per_role_conf}, confidence=min(per_role_conf.values()),
                          entropy=0.0, ambiguous=any(c < 0.6 for c in per_role_conf.values()),
                          per_role_conf=dict(per_role_conf))


def test_policy_commits_when_all_roles_confident():
    pol = ResolutionPolicy(ResolutionConfig(commit_threshold=0.6))
    a = pol.decide(_ra({"m": 0.9, "s": 0.8}), TaskContext(), 0, 0, None)
    assert a.kind == "commit"


def test_policy_inspects_then_clarifies_then_abstains():
    ra = _ra({"m": 0.4, "s": 0.9})
    # inspection allowed -> observe first
    pol = ResolutionPolicy(ResolutionConfig(commit_threshold=0.6))
    assert pol.decide(ra, TaskContext(), 0, 0, None).kind == "observe"
    # inspection exhausted -> escalate to clarification
    assert pol.decide(ra, TaskContext(), 3, 0, 0.0).kind == "ask_user"
    # clarification exhausted too -> abstain, never guess
    assert pol.decide(ra, TaskContext(), 3, 2, 0.0).kind == "abstain"


def test_policy_treats_clarified_roles_as_resolved():
    pol = ResolutionPolicy(ResolutionConfig(commit_threshold=0.6))
    ctx = TaskContext(user_selections={"m"})           # user already answered "m"
    # only s is contested now; m is trusted despite low conf
    assert pol.contested(_ra({"m": 0.1, "s": 0.9}), ctx) == []


def test_inspection_only_abstains_on_genuine_ambiguity():
    cfg = replace(ExecConfig(), resolution=True, allow_inspection=True, allow_clarification=False)
    r = aggregate(run_benchmark(seeds=range(20), cfg=cfg))
    assert r.abstention_rate > 0.0                     # can't resolve ties -> declines
    assert r.clarification_rate == 0.0                 # never asked


def test_clarification_resolves_ambiguity_without_abstaining():
    cfg = replace(ExecConfig(), resolution=True, allow_inspection=False, allow_clarification=True)
    r = aggregate(run_benchmark(seeds=range(20), cfg=cfg))
    assert r.abstention_rate == 0.0                    # a question always available
    assert r.clarification_rate > 0.0                  # it did ask
    assert r.ambiguity_resolution_rate > 0.5           # ambiguous -> mostly resolved correctly


def test_resolution_off_is_v0_3_behaviour():
    r = aggregate(run_benchmark(seeds=range(20), cfg=ExecConfig()))
    assert r.abstention_rate == 0.0
    assert r.clarification_rate == 0.0
    assert r.autonomous_coverage == 1.0


def test_committed_risk_is_decomposed_and_ordered():
    # silent (believed & failed) must be <= any-failure among committed; both are
    # measured on the SAME denominator so they are directly comparable.
    cfg = replace(ExecConfig(), resolution=True, allow_inspection=False, allow_clarification=True)
    r = aggregate(run_benchmark(seeds=range(20), cfg=cfg))
    assert r.risk_silent_committed <= r.selective_risk + 1e-9
    assert 0.0 <= r.auto_cov_identifiable <= 1.0
    # coverage on genuinely-ambiguous scenes should be far below identifiable ones
    assert r.auto_cov_ambiguous < r.auto_cov_identifiable


def test_clarification_persists_across_a_workflow():
    from osc.benchmark.runner import run_workflows
    cfg = replace(ExecConfig(), resolution=True, allow_inspection=True, allow_clarification=True)
    w = run_workflows(n_workflows=4, orders_per_workflow=15, cfg=cfg)
    # the customer is asked at setup, then almost never again in production.
    assert w["clarifications_per_production_ep"] < 0.5
    assert w["repeated_question_rate"] < 0.5
    # and the saved answer still produces working executions on new instances.
    assert w["production_success"] > 0.5


def test_paired_scenarios_each_action_resolves_its_own_case():
    from osc.benchmark.resolution_scenarios import run_scenarios
    r = run_scenarios(seeds=range(40))
    # the RIGHT capability resolves each scenario autonomously & correctly...
    assert r["noisy_identifiable"]["inspection-only"]["autonomous_correct"] > 0.8
    assert r["occluded"]["+viewpoint"]["autonomous_correct"] > 0.8
    assert r["interaction"]["+probe"]["autonomous_correct"] > 0.8
    assert r["fundamental"]["+metadata"]["autonomous_correct"] > 0.8


def test_inspection_never_fakes_fundamental_ambiguity():
    from osc.benchmark.resolution_scenarios import run_scenarios
    r = run_scenarios(seeds=range(40))
    # no physical capability may "resolve" a fundamentally ambiguous scene
    for cap in ("inspection-only", "+viewpoint", "+probe"):
        assert r["fundamental"][cap]["autonomous_correct"] < 0.1
    # only asking a human / SKU metadata resolves it
    assert r["fundamental"]["clarification"]["human"] > 0.8


def test_wrong_capability_does_not_resolve_occluded_or_interaction():
    from osc.benchmark.resolution_scenarios import run_scenarios
    r = run_scenarios(seeds=range(40))
    assert r["occluded"]["+probe"]["autonomous_correct"] < 0.1      # probe can't see a label
    assert r["interaction"]["+viewpoint"]["autonomous_correct"] < 0.1  # viewpoint can't feel mass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
