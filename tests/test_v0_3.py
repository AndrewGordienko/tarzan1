"""v0.3 semantic-gate tests: correspondence must ignore randomized appearance,
and the scorer must recognise objectively unidentifiable scenes so that a wrong
binding there is labelled ambiguous, not a silent perception failure.
"""
from types import SimpleNamespace

import numpy as np

from osc.agent.belief import BeliefObject, BeliefState
from osc.benchmark.scorer import Scorer
from osc.geometry import pose
from osc.sim.base import SimObject, SimState
from osc.skills.correspondence import correspond


def _bobj(tid, sz, color):
    return BeliefObject(track_id=tid, pose=pose(0.4, 0.0), size=np.array([sz, sz, sz]),
                        shape="box", color=color)


def test_correspondence_ignores_color_even_when_it_misleads():
    # demo signatures: manipuland ~0.036 (red), support ~0.050 (blue).
    demo = BeliefState(objects={
        "m": _bobj("m", 0.036, "red"),
        "s": _bobj("s", 0.050, "blue"),
    })
    from osc.tasks import TASKS, record_demo
    # build role signatures the way the compiler does, then hand-craft an eval
    # belief where COLOUR points the wrong way but SIZE is correct.
    graph = record_demo(TASKS["stack"])
    # eval: the correct manipuland is GREEN (not demo's red) and a distractor
    # wears the demo manipuland's RED. A colour-weighted matcher would mis-bind.
    b = BeliefState(objects={
        "t0": _bobj("t0", 0.036, "green"),      # true manipuland (right size)
        "t1": _bobj("t1", 0.050, "purple"),     # true support (right size)
        "t2": _bobj("t2", 0.031, "red"),        # distractor wearing demo's colour
    })
    corr = correspond(b, graph.role_signatures)
    assert corr["manipuland0"] == "t0"          # bound by size, not colour
    assert corr["support0"] == "t1"


def _scorer_with(role_to_gt, role_signatures):
    graph = SimpleNamespace(role_to_gt=role_to_gt, role_signatures=role_signatures)
    return Scorer(task=SimpleNamespace(name="t"), roles={}, graph=graph)


def _state(objs):
    return SimState(objects={o.name: o for o in objs}, gripper=np.zeros(4),
                    gripper_closed=False, grasped=None,
                    table_bounds=(0, 1, -0.5, 0.5), table_z=0.0, fallen=set(), t=0)


def test_identifiable_true_when_target_is_uniquely_best_match():
    # demo manipuland signature ~ size 0.036; eval has a clearly-0.036 target and
    # a far-off distractor -> the true object is the unique best match.
    sc = _scorer_with({"manipuland0": "A"},
                      {"manipuland0": np.array([0.036, 0.036, 0.0, 0.0])})
    s = _state([
        SimObject(name="A", size=np.array([0.036, 0.036, 0.036]), pose=pose(0.4, 0.0)),
        SimObject(name="D", size=np.array([0.055, 0.055, 0.055]), pose=pose(0.6, 0.0)),
    ])
    assert sc.identifiable(s) is True


def test_identifiable_false_when_a_distractor_ties_the_target():
    # the true target drifted to 0.050 while a distractor sits exactly at the
    # demonstrated 0.036 -> no size-based agent could pick the right one.
    sc = _scorer_with({"manipuland0": "A"},
                      {"manipuland0": np.array([0.036, 0.036, 0.0, 0.0])})
    s = _state([
        SimObject(name="A", size=np.array([0.050, 0.050, 0.050]), pose=pose(0.4, 0.0)),
        SimObject(name="D", size=np.array([0.036, 0.036, 0.036]), pose=pose(0.6, 0.0)),
    ])
    assert sc.identifiable(s) is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
