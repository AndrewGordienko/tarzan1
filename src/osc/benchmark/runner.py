"""Benchmark runner: evaluation splits x tasks x seed groups -> reports.

Splits isolate factors (each changes one thing vs the base):
  seen_task_new_layout : layout jitter + distractors, mild perception/dynamics
  unseen_instances     : role object sizes perturbed, distractor shapes random
  hidden_dynamics      : wide friction/mass/actuator-delay ranges
  disturbance_recovery : base + one recoverable disturbance on the manipuland

Every episode runs on BeliefState through AgentEnv; ground truth is read only by
the Scorer afterward. Emits JSON + Markdown + a failure-taxonomy file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from ..agent.dynamics_context import DynamicsContext
from ..agent.env import AgentEnv
from ..agent.estimator import StateEstimator
from ..execution.loop import ClosedLoopExecutor, ExecConfig
from ..perception.detections import CorruptionSpec
from ..sim.disturbance import sample_disturbance
from ..sim.randomize import RandomizationSpec
from ..tasks import DEFAULT_TASKS, record_demo
from ..worldmodel.planning_model import PlanningModel
from .scorer import Scorer


@dataclass
class Split:
    name: str
    rand: RandomizationSpec
    corr: CorruptionSpec
    disturb: bool = False


def default_splits() -> list[Split]:
    base_corr = CorruptionSpec(pos_noise=0.004, occlusion_prob=0.04, drop_prob=0.02,
                               delay_frames=0, false_contact_prob=0.02, identity_swap_prob=0.01)
    return [
        Split("seen_task_new_layout", RandomizationSpec(), base_corr),
        Split("unseen_instances",
              RandomizationSpec(role_size_jitter=0.25, n_distractors=3), base_corr),
        Split("hidden_dynamics",
              RandomizationSpec(friction_range=(0.2, 1.1), mass_range=(0.03, 0.6),
                                actuator_delay_range=(0.1, 0.6)), base_corr),
        Split("disturbance_recovery", RandomizationSpec(), base_corr, disturb=True),
    ]


def run_episode(task, graph, split: Split, seed: int, cfg: ExecConfig,
                privileged: bool = False, task_context=None,
                oracle_role_binding: bool = False):
    from ..sim.randomize import randomize
    state, backend, roles = randomize(task.scene, split.rand, seed=seed)
    disturbance = None
    if split.disturb:
        horizon = max(12, task.max_total_steps // 20)
        disturbance = sample_disturbance(["manip"], horizon=horizon, seed=seed)
        backend._pre_step_hook = disturbance
    backend.reset(state)

    env = AgentEnv(backend, split.corr, rng=np.random.default_rng(seed + 101))
    est = _estimator(backend, privileged)
    ctx = DynamicsContext()
    pm = PlanningModel(ctx, table_bounds_est=state.table_bounds)
    clarify_fn = _clarify_fn(graph, backend) if cfg.resolution and cfg.allow_clarification else None
    oracle_corr = _oracle_corr_fn(graph, backend) if oracle_role_binding else None
    execu = ClosedLoopExecutor(env, graph, est, ctx, pm,
                               ExecConfig(**{**cfg.__dict__,
                                             "max_total_steps": task.max_total_steps}),
                               clarify_fn=clarify_fn, task_context=task_context,
                               oracle_corr=oracle_corr)
    trace = execu.run()

    final = backend.state()
    manip = state.objects["manip"]
    true_params = (backend.actuator_delay, manip.friction / 0.6, manip.mass / 0.1)
    rec = Scorer(task, roles, graph).score(split.name, seed, final, env.step_info_log,
                                    trace, disturbance, ctx, true_params, initial_state=state)
    return rec


def run_workflow(task, graph, split: Split, seeds, cfg: ExecConfig):
    """One WORKFLOW = a persistent TaskContext shared across many production orders
    (new layouts, track ids, object instances each seed). A clarification answered
    on the first order must carry to the rest -- the product asks once at setup,
    then runs thousands of boxes untouched. Returns (records, per-workflow stats)."""
    from ..execution.resolution import TaskContext
    ctx = TaskContext()
    seeds = list(seeds)
    records = [run_episode(task, graph, split, s, cfg, task_context=ctx) for s in seeds]
    prod = records[1:]                      # everything after the first (setup) order
    n_prod = max(1, len(prod))
    stats = dict(
        clarifications_per_workflow=sum(r.clarifications for r in records),
        clarifications_setup=records[0].clarifications,
        clarifications_per_production_ep=sum(r.clarifications for r in prod) / n_prod,
        repeated_question_rate=float(np.mean([r.clarifications > 0 for r in prod])) if prod else 0.0,
        production_role_accuracy=float(np.mean([r.role_binding_correct for r in prod])) if prod else 0.0,
        production_success=float(np.mean([r.success for r in prod])) if prod else 0.0,
    )
    return records, stats


def run_workflows(task=None, split=None, n_workflows=8, orders_per_workflow=20,
                  cfg: ExecConfig | None = None):
    """Aggregate run_workflow over several distinct workflows (disjoint seed blocks)."""
    task = task or DEFAULT_TASKS[0]
    split = split or default_splits()[1]     # unseen_instances: where ambiguity concentrates
    cfg = cfg or ExecConfig()
    graph = record_demo(task)
    per_wf = []
    for w in range(n_workflows):
        base = 1000 * (w + 1)
        _, stats = run_workflow(task, graph, split, range(base, base + orders_per_workflow), cfg)
        per_wf.append(stats)
    keys = per_wf[0].keys()
    return {k: float(np.mean([s[k] for s in per_wf])) for k in keys}


def _clarify_fn(graph, backend):
    """The 'user': answers which track plays a queried role by pointing at the
    correct GT object (nearest track). Models a customer clicking the object /
    SKU metadata -- information the demonstration alone did not contain. Only the
    ASKED roles are answered; the agent never reads this itself."""
    from ..geometry import dist_xyz
    r2g = getattr(graph, "role_to_gt", {})
    def fn(target_roles, belief):
        out, gt = {}, backend.state()
        for role in target_roles:
            gt_name = r2g.get(role)
            if gt_name is None or gt_name not in gt.objects or not belief.objects:
                continue
            gp = gt.objects[gt_name].pose
            out[role] = min(sorted(belief.objects),
                            key=lambda t: dist_xyz(belief.objects[t].pose, gp))
        return out
    return fn


def _oracle_corr_fn(graph, backend):
    """Upper-bound role binding only.  Kept distinct from `privileged`, whose
    oracle estimator still exercises the normal correspondence implementation."""
    from ..geometry import dist_xyz
    r2g = getattr(graph, "role_to_gt", {})
    def fn(belief):
        gt = backend.state()
        return {role: min(sorted(belief.objects),
                          key=lambda tid: dist_xyz(belief.objects[tid].pose,
                                                   gt.objects[name].pose))
                for role, name in r2g.items()
                if name in gt.objects and belief.objects}
    return fn


def _estimator(backend, privileged: bool):
    """Full system: estimate from percepts. Ablation `privileged`: an oracle
    estimator that reads ground truth (upper bound; used only in ablations)."""
    if not privileged:
        return StateEstimator()
    from .oracle_estimator import OracleEstimator
    return OracleEstimator(backend)


def run_benchmark(tasks=None, splits=None, seeds=range(20), cfg: ExecConfig | None = None,
                  privileged: bool = False, graphs: dict | None = None,
                  oracle_role_binding: bool = False):
    tasks = tasks or list(DEFAULT_TASKS)
    splits = splits or default_splits()
    cfg = cfg or ExecConfig()
    # A camera-enabled deployment must compile an equally camera-enabled demo.
    # Cache by camera mode so the same compiled evidence is reused within a block.
    graph_cache = dict(graphs or {})
    records = []
    for split in splits:
        camera_model = bool(getattr(split.rand, "camera_model", False))
        for t in tasks:
            key = (t.name, camera_model)
            graph = graph_cache.get(key)
            if graph is None and not camera_model:
                graph = graph_cache.get(t.name)
            if graph is None:
                graph = record_demo(t, camera_model=camera_model)
                graph_cache[key] = graph
            for s in seeds:
                records.append(run_episode(t, graph, split, int(s), cfg,
                                           privileged=privileged,
                                           oracle_role_binding=oracle_role_binding))
    return records


# ---------------------------------------------------------------- outputs
def _json_default(o):
    import numpy as np
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_reports(records, report, out_prefix: str):
    with open(out_prefix + ".json", "w") as f:
        json.dump({"summary": _report_dict(report),
                   "episodes": [asdict(r) for r in records]}, f, indent=2,
                  default=_json_default)
    with open(out_prefix + ".md", "w") as f:
        f.write(_markdown(report))
    fails = [r for r in records if not r.success]
    with open(out_prefix + ".failures.jsonl", "w") as f:
        for r in fails:
            f.write(json.dumps({"task": r.task, "split": r.split, "seed": r.seed,
                                "category": r.failure_category,
                                "wrong_belief": r.wrong_belief,
                                "replans": r.autonomous_replans,
                                "collisions": r.collisions,
                                "irreversible": r.irreversible_failures,
                                "steps": r.steps}) + "\n")
    return out_prefix


def _report_dict(report):
    d = report.__dict__.copy()
    return d


def _markdown(report) -> str:
    r = report
    lines = ["# OSC v0.3 Benchmark Report", "",
             f"- episodes: **{r.n}**",
             f"- success: **{r.success_rate:.1%}** (CI95 {r.success_ci[0]:.2f}–{r.success_ci[1]:.2f})",
             f"- first-attempt success: **{r.first_attempt_rate:.1%}**",
             f"- wrong-belief (silent error): **{r.wrong_belief_rate:.1%}**",
             f"- recovery: {r.recovered}/{r.recovery_opportunities} opportunities ({r.recovery_rate:.0%})",
             f"- autonomous replans: {r.autonomous_replans_total}; human interventions: {r.human_interventions_total}",
             f"- plan latency ms p50/p95/p99: {r.plan_latency_ms['p50']:.1f}/{r.plan_latency_ms['p95']:.1f}/{r.plan_latency_ms['p99']:.1f}",
             f"- sensor→action ms p50/p95: {r.step_latency_ms['p50']:.2f}/{r.step_latency_ms['p95']:.2f}",
             f"- completion steps / sim-sec: {r.mean_completion_steps:.0f} / {r.mean_completion_seconds:.2f}",
             f"- safety violations/ep: {r.safety_violations_per_ep:.2f} (collisions/ep {r.collisions_per_ep:.2f})",
             f"- failure breakdown: `{r.failure_breakdown}`",
             f"- context est. error (mean abs): `{r.context_error}`", "",
             "## By split", "", "| split | success | CI95 | n |", "|---|---|---|---|"]
    for sp, d in r.by_split.items():
        lines.append(f"| {sp} | {d['success_rate']:.1%} | {d['ci'][0]:.2f}–{d['ci'][1]:.2f} | {d['n']} |")
    return "\n".join(lines) + "\n"
