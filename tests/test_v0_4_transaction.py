"""Clarification as a GLOBAL one-to-one assignment transaction: pinning a role
recomputes the whole assignment, every role is re-checked, and commit happens only
when the entire assignment is supported -- not merely because one question was asked.
"""
import numpy as np

from osc.agent.belief import BeliefObject, BeliefState
from osc.execution.resolution import ResolutionConfig, ResolutionPolicy, TaskContext
from osc.geometry import pose
from osc.skills.correspondence import RoleBelief
from osc.tasks import TASKS, record_demo


def _o(tid, sz, contested=False):
    return BeliefObject(track_id=tid, pose=pose(0.4, 0.0), size=np.array([sz, sz, sz]),
                        shape="box", color="x", association_contested=contested)


def _rb():
    return RoleBelief(record_demo(TASKS["stack"]).role_signatures)


def test_pin_sets_confidence_and_excludes_track():
    b = BeliefState(objects={"t0": _o("t0", 0.036), "t1": _o("t1", 0.050), "t3": _o("t3", 0.049)})
    ra = _rb().update(b, fixed={"manipuland0": "t0"})
    assert ra.per_role_conf["manipuland0"] == 1.0
    assert ra.mapping["manipuland0"] == "t0"
    assert ra.mapping["support0"] != "t0"          # pinned track removed from the pool


def test_pinning_one_role_changes_another_via_one_to_one():
    b = BeliefState(objects={"t0": _o("t0", 0.036), "t1": _o("t1", 0.050)})
    rb = _rb()
    base = rb.update(b).mapping                     # manip->t0, support->t1
    forced = rb.update(b, fixed={"manipuland0": "t1"}).mapping
    assert base["support0"] == "t1"
    assert forced["support0"] == "t0"              # pinning manip->t1 pushed support to t0


def test_clarifying_one_role_leaves_still_ambiguous_role_contested():
    # two support-sized tracks make support genuinely contested; clarifying the
    # (already-confident) manipuland must NOT make the assignment committable.
    b = BeliefState(objects={"t0": _o("t0", 0.036), "t1": _o("t1", 0.050), "t3": _o("t3", 0.049)})
    pol = ResolutionPolicy(ResolutionConfig(commit_threshold=0.60))
    ctx = TaskContext(user_selections={"manipuland0"})
    ra = _rb().update(b, fixed={"manipuland0": "t0"})
    assert not pol.committable(ra, ctx)            # support still contested -> ask again
    assert pol.decide(ra, ctx, 3, 0, 0.0).kind == "ask_user"
    assert pol.decide(ra, ctx, 3, 0, 0.0).target_roles == ("support0",)


def test_committable_only_when_every_role_supported():
    b = BeliefState(objects={"t0": _o("t0", 0.036), "t1": _o("t1", 0.050), "t3": _o("t3", 0.049)})
    pol = ResolutionPolicy(ResolutionConfig(commit_threshold=0.60))
    ctx = TaskContext(user_selections={"manipuland0", "support0"})
    ra = _rb().update(b, fixed={"manipuland0": "t0", "support0": "t1"})
    assert pol.committable(ra, ctx)                # both pinned -> whole assignment supported
    assert pol.decide(ra, ctx, 0, 1, None).kind == "commit"


def test_association_contest_blocks_commit_even_with_a_confident_assignment():
    b = BeliefState(objects={"t0": _o("t0", 0.036, contested=True),
                              "t1": _o("t1", 0.050)})
    ra = _rb().update(b)
    pol = ResolutionPolicy(ResolutionConfig(allow_inspection=False, allow_clarification=False,
                                            commit_threshold=0.60))
    assert "manipuland0" in ra.association_contested
    assert not pol.committable(ra, TaskContext())
    assert pol.decide(ra, TaskContext(), 3, 0, 0.0).kind == "abstain"


def test_workflow_asks_at_setup_then_stops_repeating():
    from dataclasses import replace
    from osc.benchmark.runner import run_workflows
    from osc.execution.loop import ExecConfig
    cfg = replace(ExecConfig(), resolution=True, allow_inspection=False, allow_clarification=True)
    w = run_workflows(n_workflows=4, orders_per_workflow=15, cfg=cfg)
    assert w["clarifications_per_production_ep"] < 0.5   # not re-asking every box
    assert w["repeated_question_rate"] < 0.5


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
