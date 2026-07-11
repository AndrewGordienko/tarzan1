"""Paired oracle attribution ladder.

Replays every episode (same task/split/seed) under configurations that swap in a
perfect version of one component at a time, so the success gain from fixing each
is measured, not assumed. This is what turns the circular "perception" label into
an evidence-based error budget.

Oracle components (benchmark-side; the deployed agent never uses these):
  * oracle_tracks         : OracleEstimator (ground-truth poses, no noise)
  * oracle_correspondence : bind roles to the correct GT objects directly
  * oracle_verifier       : stop/declare success from ground truth
  * full_oracle           : all of the above

Semantic-target and controller oracles are added once semantic retargeting lands
(they need the constraint solver to isolate target-pose error from control).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agent.dynamics_context import DynamicsContext
from ..agent.env import AgentEnv
from ..agent.estimator import StateEstimator
from ..execution.loop import ClosedLoopExecutor, ExecConfig
from ..geometry import dist_xyz
from ..worldmodel.planning_model import PlanningModel
from .oracle_estimator import OracleEstimator
from .runner import default_splits
from .scorer import Scorer


@dataclass
class LadderConfig:
    name: str
    oracle_tracks: bool = False
    oracle_corr: bool = False
    oracle_verifier: bool = False


LADDER = [
    LadderConfig("full"),
    LadderConfig("oracle_tracks", oracle_tracks=True),
    LadderConfig("oracle_correspondence", oracle_corr=True),
    LadderConfig("oracle_tracks+corr", oracle_tracks=True, oracle_corr=True),
    LadderConfig("oracle_verifier", oracle_verifier=True),
    LadderConfig("full_oracle", oracle_tracks=True, oracle_corr=True, oracle_verifier=True),
]


def _oracle_corr_fn(graph, backend):
    """Bind each role to the eval track nearest the correct GT object."""
    def fn(belief):
        out = {}
        gt = backend.state()
        for role, gt_name in sorted(graph.role_to_gt.items()):
            if gt_name not in gt.objects or not belief.objects:
                continue
            gp = gt.objects[gt_name].pose
            tid = min(sorted(belief.objects), key=lambda t: dist_xyz(belief.objects[t].pose, gp))
            out[role] = tid
        return out
    return fn


def _oracle_goal_fn(task, roles, backend):
    def fn(belief):
        return bool(task.success(backend.state(), roles))
    return fn


def run_ladder_episode(task, graph, split, seed, lc: LadderConfig):
    from ..sim.randomize import randomize
    from ..sim.disturbance import sample_disturbance
    state, backend, roles = randomize(task.scene, split.rand, seed=seed)
    disturbance = None
    if split.disturb:
        horizon = max(12, task.max_total_steps // 20)
        disturbance = sample_disturbance(["manip"], horizon=horizon, seed=seed)
        backend._pre_step_hook = disturbance
    backend.reset(state)

    env = AgentEnv(backend, split.corr, rng=np.random.default_rng(seed + 101))
    est = OracleEstimator(backend) if lc.oracle_tracks else StateEstimator()
    ctx = DynamicsContext()
    pm = PlanningModel(ctx, table_bounds_est=state.table_bounds)
    execu = ClosedLoopExecutor(
        env, graph, est, ctx, pm, ExecConfig(max_total_steps=task.max_total_steps),
        oracle_corr=_oracle_corr_fn(graph, backend) if lc.oracle_corr else None,
        oracle_goal=_oracle_goal_fn(task, roles, backend) if lc.oracle_verifier else None)
    trace = execu.run()

    final = backend.state()
    manip = state.objects["manip"]
    true_params = (backend.actuator_delay, manip.friction / 0.6, manip.mass / 0.1)
    rec = Scorer(task, roles, graph).score(f"{split.name}", seed, final, env.step_info_log,
                                    trace, disturbance, ctx, true_params, initial_state=state)
    return rec


def run_ladder(tasks=None, splits=None, seeds=range(20)):
    from ..tasks import DEFAULT_TASKS, record_demo
    tasks = tasks or list(DEFAULT_TASKS)
    splits = splits or default_splits()
    graphs = {t.name: record_demo(t) for t in tasks}
    rows = {}
    for lc in LADDER:
        recs = []
        for split in splits:
            for t in tasks:
                for s in seeds:
                    recs.append(run_ladder_episode(t, graphs[t.name], split, int(s), lc))
        rows[lc.name] = recs
    return rows


def error_budget(rows: dict) -> str:
    order = [lc.name for lc in LADDER]
    succ = {k: np.mean([r.success for r in v]) for k, v in rows.items()}
    wb = {k: np.mean([r.wrong_belief for r in v]) for k, v in rows.items()}
    lines = ["ERROR-BUDGET LADDER (paired seeds; success rate under each oracle swap)",
             f"{'config':26s} {'success':>8s} {'Δ vs full':>10s} {'wrong_belief':>13s}",
             "-" * 60]
    base = succ["full"]
    for k in order:
        lines.append(f"{k:26s} {succ[k]:7.1%} {succ[k]-base:+9.1%} {wb[k]:12.1%}")
    lines.append("-" * 60)
    lines.append("Attribution (marginal success gain from each perfect component):")
    lines.append(f"  state estimation        : {succ['oracle_tracks']-base:+.1%}")
    lines.append(f"  role correspondence     : {succ['oracle_correspondence']-base:+.1%}")
    lines.append(f"  perception+binding      : {succ['oracle_tracks+corr']-base:+.1%}")
    lines.append(f"  verifier (stop timing)  : {succ['oracle_verifier']-base:+.1%}")
    lines.append(f"  full oracle upper bound : {succ['full_oracle']-base:+.1%} "
                 f"(={succ['full_oracle']:.1%}); residual to 100% = planning/control/target")
    return "\n".join(lines)
