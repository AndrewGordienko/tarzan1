"""v0.2.1 truth-harness repair tests.

These pin the properties the v0.2.1 pass is about: the physics that ground-truth
scoring depends on is correct, safety events are structurally *reachable* (not
guaranteed-zero), disturbances only count when they truly perturb the world, and
the whole benchmark is deterministic ACROSS processes (not just within one).
"""
import json
import os
import subprocess
import sys

import numpy as np

from osc.benchmark.scorer import gt_on_top
from osc.geometry import pose
from osc.sim.base import Action, SimObject, SimState
from osc.sim.disturbance import Disturbance
from osc.sim.toy import ToyTabletopSim


def _state(objects, grasped=None):
    return SimState(objects={o.name: o for o in objects}, gripper=np.zeros(4),
                    gripper_closed=False, grasped=grasped,
                    table_bounds=(0.0, 1.0, -0.5, 0.5), table_z=0.0, fallen=set(), t=0)


# --- physics: a stack settles in contact, and gt_on_top demands contact -------
def test_stack_settles_in_contact_not_floating():
    support = SimObject(name="B", size=np.array([0.05, 0.05, 0.06]),
                        pose=np.array([0.4, 0.0, 0.0, 0.0]))
    mover = SimObject(name="A", size=np.array([0.04, 0.04, 0.04]),
                      pose=np.array([0.4, 0.0, 0.30, 0.0]))
    s = _state([support, mover])
    sim = ToyTabletopSim(); sim._s = s
    sim._settle()
    gap = (mover.pose[2] - mover.size[2] / 2) - (support.pose[2] + support.size[2] / 2)
    assert abs(gap) < 1e-6, f"stacked object floats/sinks by {gap:.4f} m"
    assert gt_on_top(s, "A", "B")


def test_gt_on_top_rejects_a_floating_object():
    support = SimObject(name="B", size=np.array([0.05, 0.05, 0.06]),
                        pose=np.array([0.4, 0.0, 0.03, 0.0]))       # top at 0.06
    floating = SimObject(name="A", size=np.array([0.04, 0.04, 0.04]),
                         pose=np.array([0.4, 0.0, 0.20, 0.0]))      # base at 0.18
    s = _state([support, floating])
    assert not gt_on_top(s, "A", "B")


# --- safety: irreversible + force violations are REACHABLE --------------------
def test_irreversible_set_when_object_leaves_table():
    o = SimObject(name="A", size=np.array([0.04, 0.04, 0.04]),
                  pose=np.array([0.99, 0.0, 0.02, 0.0]))            # at the edge
    s = _state([o])
    sim = ToyTabletopSim(); sim._s = s
    # command the gripper (no grasp) — just step so _settle runs with info.
    _, info = sim.step(Action(target=np.array([0.99, 0.0, 0.02, 0.0]), gripper_close=0.0))
    # move it off the edge and settle again
    o.pose[0] = 1.5
    _, info = sim.step(Action(target=np.array([0.99, 0.0, 0.02, 0.0]), gripper_close=0.0))
    assert "A" in s.fallen
    assert info.irreversible is True


def test_force_violation_reachable_within_configured_mass_range():
    # heaviest configured object (mass 0.40, friction 0.90) driven hard laterally.
    # The discrete contact model checks the TCP's END position, so place the object
    # at the motion endpoint: the TCP arrives there fast == a hard shove.
    delay = 0.25
    start, target = 0.30, 0.72
    endpoint = start + (1.0 - delay) * (target - start)    # first-order-lag TCP
    heavy = SimObject(name="H", size=np.array([0.05, 0.05, 0.05]), mass=0.40,
                      friction=0.90, pose=np.array([endpoint, 0.0, 0.05, 0.0]))
    s = _state([heavy])
    s.gripper = np.array([start, 0.0, 0.05, 0.0])
    sim = ToyTabletopSim(actuator_delay=delay); sim._s = s
    _, info = sim.step(Action(target=np.array([target, 0.0, 0.05, 0.0]), gripper_close=0.0))
    assert info.collision is True
    assert info.force_violation is True


def test_gentle_contact_is_not_a_force_violation():
    heavy = SimObject(name="H", size=np.array([0.05, 0.05, 0.05]), mass=0.40,
                      friction=0.90, pose=np.array([0.50, 0.0, 0.05, 0.0]))
    s = _state([heavy])
    s.gripper = np.array([0.47, 0.0, 0.05, 0.0])
    sim = ToyTabletopSim(actuator_delay=0.25); sim._s = s
    # a small nudge toward the object: contact, but not a hard drive-in.
    _, info = sim.step(Action(target=np.array([0.485, 0.0, 0.05, 0.0]), gripper_close=0.0))
    assert info.force_violation is False


# --- disturbance only counts when it actually perturbs the world --------------
def test_drop_disturbance_on_unheld_object_is_a_noop():
    o = SimObject(name="A", size=np.array([0.04, 0.04, 0.04]),
                  pose=np.array([0.4, 0.0, 0.02, 0.0]))
    s = _state([o], grasped=None)                # nothing is held
    from osc.sim.base import StepInfo
    d = Disturbance("drop", "A", at_step=0, magnitude=0.05, rng=np.random.default_rng(0))
    d(s, StepInfo())
    assert d.fired is True
    assert d.perturbed is False                  # a drop with nothing held changes nothing


def test_displace_disturbance_perturbs():
    o = SimObject(name="A", size=np.array([0.04, 0.04, 0.04]),
                  pose=np.array([0.4, 0.0, 0.02, 0.0]))
    s = _state([o])
    from osc.sim.base import StepInfo
    before = o.pose[:2].copy()
    d = Disturbance("displace", "A", at_step=0, magnitude=0.05, rng=np.random.default_rng(0))
    d(s, StepInfo())
    assert d.perturbed is True
    assert np.linalg.norm(o.pose[:2] - before) > 0.01


# --- determinism ACROSS processes (not just within one) -----------------------
_VOLATILE = {"plan_latency_ms", "step_latency_ms"}   # wall-clock, excluded by design


def _bench_summary(hashseed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH="src")
    out = f"/tmp/osc_dettest_{hashseed}"
    subprocess.run([sys.executable, "-m", "osc.run_bench", "--seeds", "8", "--out", out],
                   check=False, env=env, capture_output=True)
    with open(out + ".json") as f:
        return json.load(f)["summary"]


def test_benchmark_is_deterministic_across_processes():
    a = _bench_summary("0")
    b = _bench_summary("12345")
    a = {k: v for k, v in a.items() if k not in _VOLATILE}
    b = {k: v for k, v in b.items() if k not in _VOLATILE}
    assert a == b, "benchmark summary differs across PYTHONHASHSEED (non-determinism)"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
