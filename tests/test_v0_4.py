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


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
