"""v0.2 behavioural + architectural tests.

These enforce the properties the v0.2 upgrade is *about*: no privileged-state
access on the agent path, name-independent correspondence, one-shot transfer on
belief state, an honest no-weight-update guarantee, and relative>absolute.
"""
import sys

import numpy as np
import pytest

from osc.agent.belief import BeliefObject, BeliefState
from osc.agent.env import AgentEnv, PrivilegedAccessError
from osc.agent.estimator import StateEstimator
from osc.agent.dynamics_context import DynamicsContext
from osc.benchmark.runner import run_benchmark, Split
from osc.execution.loop import ClosedLoopExecutor, ExecConfig
from osc.geometry import pose
from osc.metrics.metrics import aggregate
from osc.perception.detections import CorruptionSpec
from osc.sim.randomize import RandomizationSpec, randomize
from osc.skills.correspondence import correspond
from osc.tasks import TASKS, record_demo
from osc.worldmodel.planning_model import PlanningModel


# --- 1. architecture: the agent path must never read ground-truth state -----
def test_agent_env_forbids_privileged_state():
    task = TASKS["stack"]
    state, backend, _ = randomize(task.scene, RandomizationSpec(), seed=1)
    backend.reset(state)
    env = AgentEnv(backend, CorruptionSpec(), rng=np.random.default_rng(1))
    with pytest.raises(PrivilegedAccessError):
        env.state()
    with pytest.raises(PrivilegedAccessError):
        _ = env.objects


def test_execution_does_not_touch_ground_truth():
    """Run a full episode with backend.state() booby-trapped: if any agent code
    calls it, the episode raises. It must complete."""
    task = TASKS["stack"]
    graph = record_demo(task)
    state, backend, _ = randomize(task.scene, RandomizationSpec(), seed=2)
    backend.reset(state)
    calls = {"n": 0}
    real_state = backend.state

    def tripwire(*a, **k):
        calls["n"] += 1
        raise AssertionError("agent path read ground-truth SimState")
    backend.state = tripwire

    env = AgentEnv(backend, CorruptionSpec(), rng=np.random.default_rng(2))
    est = StateEstimator(); ctx = DynamicsContext()
    pm = PlanningModel(ctx, table_bounds_est=state.table_bounds)
    trace = ClosedLoopExecutor(env, graph, est, ctx, pm, ExecConfig()).run()
    assert calls["n"] == 0
    assert trace.steps > 0
    backend.state = real_state          # scorer may use it afterward


# --- 2. name independence: correspondence picks by geometry, not name/order --
def test_correspondence_ignores_names_and_order_with_distractors():
    task = TASKS["stack"]
    graph = record_demo(task)
    # a belief with the true manipuland/target sizes plus two distractors, in
    # arbitrary track-id order and with unrelated names.
    def obj(tid, sz):
        return BeliefObject(track_id=tid, pose=pose(np.random.rand(), np.random.rand()),
                            size=np.array([sz, sz, sz]), shape="box", color="zzz")
    b = BeliefState(objects={
        "zX": obj("zX", 0.031),          # distractor
        "q7": obj("q7", 0.050),          # matches target size
        "aa": obj("aa", 0.036),          # matches manipuland size
        "m2": obj("m2", 0.045),          # distractor
    })
    corr = correspond(b, graph.role_signatures)
    assert corr["manipuland0"] == "aa"   # smallest ~ demo manipuland
    assert corr["support0"] == "q7"      # ~ demo target


# --- 3. one-shot transfer on belief state, no fine-tuning -------------------
def test_one_shot_transfer_on_belief():
    clean = [Split("clean", RandomizationSpec(n_distractors=0), CorruptionSpec(enabled=False))]
    for name in ("stack", "side_place"):
        rep = aggregate(run_benchmark(tasks=[TASKS[name]], splits=clean, seeds=range(12)))
        assert rep.success_rate >= 0.8, f"{name} only {rep.success_rate:.2f}"


# --- 4. honest no-weight-update guarantee ----------------------------------
def test_no_persistent_learning_across_episodes():
    """The compiled program is not mutated by evaluation, and repeating a seed
    reproduces the outcome exactly (=> no hidden cross-episode accumulation)."""
    task = TASKS["stack"]
    graph = record_demo(task)
    before = graph.pretty()
    split = Split("s", RandomizationSpec(), CorruptionSpec())

    from osc.benchmark.runner import run_episode
    r1 = run_episode(task, graph, split, 5, ExecConfig())
    r2 = run_episode(task, graph, split, 5, ExecConfig())
    assert graph.pretty() == before          # program unchanged by eval
    assert r1.success == r2.success and r1.steps == r2.steps   # deterministic


def test_no_optimizer_in_deployment_path():
    run_benchmark(seeds=range(2))
    # no training frameworks are imported at eval time
    assert "torch" not in sys.modules
    assert "jax" not in sys.modules


# --- 5. relative transforms beat absolute ----------------------------------
def test_relative_beats_absolute():
    from osc.benchmark.ablations import _absolute_transform_graphs
    # clean split isolates the transform frame from perception/dynamics noise, so
    # the relative>absolute effect is large and stable.
    clean = [Split("clean", RandomizationSpec(n_distractors=0), CorruptionSpec(enabled=False))]
    seeds = range(12)
    rel = aggregate(run_benchmark(splits=clean, seeds=seeds))
    ab = aggregate(run_benchmark(splits=clean, seeds=seeds,
                                 graphs=_absolute_transform_graphs()))
    assert rel.success_rate > ab.success_rate + 0.2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
