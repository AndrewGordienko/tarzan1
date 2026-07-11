"""Aggregation of EpisodeRecords into a report, with corrected definitions.

Key fixes vs v0.1:
  * latency percentiles are over INDIVIDUAL planning calls (pooled), not per-
    episode means; step (sensor->action) latency reported separately,
  * "human interventions" is a distinct field, always 0 here (no rescue API);
    autonomous replans are reported as autonomous replans,
  * recovery rate = recovered / recovery_opportunities (real opportunities only),
  * failures are categorized (perception/planning/control/irreversible/timeout),
  * completion time in control steps and simulated seconds (documented schedule),
  * bootstrap confidence intervals on success.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..benchmark.scorer import EpisodeRecord, CONTROL_HZ


def _pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def bootstrap_ci(bools, iters=2000, seed=0):
    if not bools:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(bools, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(iters, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


@dataclass
class Report:
    n: int
    success_rate: float
    success_ci: tuple
    first_attempt_rate: float
    wrong_belief_rate: float
    silent_false_completion_rate: float   # P(fail | CONFIDENT believed success)
    uncertain_completion_rate: float      # completions the agent flagged low-confidence
    p_success_given_correct_role: float   # conditional metric
    role_binding_accuracy: float
    recovery_opportunities: int
    recovered: int
    recovery_rate: float
    autonomous_replans_total: int
    human_interventions_total: int
    plan_latency_ms: dict            # p50/p95/p99/mean over pooled calls
    step_latency_ms: dict            # sensor->action, pooled
    disturbance_to_correction_steps: dict   # -> first corrective motor action
    disturbance_to_replan_steps: dict       # -> first replan
    mean_completion_steps: float
    mean_completion_seconds: float
    collisions_per_ep: float
    force_violations_per_ep: float
    irreversible_per_ep: float
    safety_violations_per_ep: float
    failure_breakdown: dict
    context_error: dict
    # -- resolution layer (autonomy vs safety) --
    selective_risk: float = 0.0            # P(fail | robot committed)
    autonomous_coverage: float = 0.0       # P(committed AND no clarification)
    abstention_rate: float = 0.0           # P(declined to act)
    clarification_rate: float = 0.0        # user questions per episode
    ambiguity_resolution_rate: float = 0.0 # initially-ambiguous -> committed & correct
    post_resolution_role_accuracy: float = 0.0   # role accuracy among committed
    unnecessary_question_rate: float = 0.0 # asked despite not being contested
    inspection_frames_per_ep: float = 0.0
    by_split: dict = field(default_factory=dict)

    def pretty(self) -> str:
        L = ["=" * 62,
             "  OSC v0.3 BENCHMARK REPORT  (belief-state, ground-truth scored)",
             "=" * 62,
             f"  episodes                      : {self.n}",
             f"  success rate                  : {self.success_rate:6.1%}  "
             f"CI95[{self.success_ci[0]:.2f},{self.success_ci[1]:.2f}]",
             f"  first-attempt success         : {self.first_attempt_rate:6.1%}",
             f"  wrong-belief rate (any)       : {self.wrong_belief_rate:6.1%}",
             f"  SILENT false-completion (conf): {self.silent_false_completion_rate:6.1%}  "
             f"(uncertain completions flagged: {self.uncertain_completion_rate:.1%})",
             f"  RESOLUTION  sel-risk/auto-cov : {self.selective_risk:6.1%} / "
             f"{self.autonomous_coverage:.1%}  abstain={self.abstention_rate:.1%} "
             f"clar/ep={self.clarification_rate:.2f} amb-resolved={self.ambiguity_resolution_rate:.1%}",
             f"  role-binding accuracy         : {self.role_binding_accuracy:6.1%}  "
             f"P(success|correct role)={self.p_success_given_correct_role:.1%}",
             f"  recovery opportunities        : {self.recovery_opportunities}"
             f"  -> recovered {self.recovered}  ({self.recovery_rate:.0%})",
             f"  autonomous replans (total)    : {self.autonomous_replans_total}",
             f"  human interventions (total)   : {self.human_interventions_total}",
             f"  plan latency ms  p50/p95/p99  : {self.plan_latency_ms['p50']:.1f} / "
             f"{self.plan_latency_ms['p95']:.1f} / {self.plan_latency_ms['p99']:.1f}",
             f"  sensor->action ms p50/p95     : {self.step_latency_ms['p50']:.2f} / "
             f"{self.step_latency_ms['p95']:.2f}",
             f"  disturbance->correction steps : action p50="
             f"{self.disturbance_to_correction_steps.get('p50',0):.0f} "
             f"(n={self.disturbance_to_correction_steps.get('n',0)}) / "
             f"replan p50={self.disturbance_to_replan_steps.get('p50',0):.0f} "
             f"(n={self.disturbance_to_replan_steps.get('n',0)})",
             f"  completion  steps / sim-sec   : {self.mean_completion_steps:.0f} / "
             f"{self.mean_completion_seconds:.2f}",
             f"  collisions / ep               : {self.collisions_per_ep:.2f}",
             f"  safety violations / ep        : {self.safety_violations_per_ep:.2f}",
             f"  failure breakdown             : {self.failure_breakdown}",
             f"  context est. error (mean abs) : "
             + ", ".join(f"{k}={v:.3f}" for k, v in self.context_error.items()),
             "=" * 62]
        if self.by_split:
            L.append("  by split:")
            for sp, d in self.by_split.items():
                L.append(f"    {sp:24s} succ={d['success_rate']:.2f} "
                         f"CI[{d['ci'][0]:.2f},{d['ci'][1]:.2f}] n={d['n']}")
            L.append("=" * 62)
        return "\n".join(L)


def _agg(records: list[EpisodeRecord], seed=0) -> dict:
    n = len(records)
    if n == 0:
        return {}
    succ = [r.success for r in records]
    # confident completions: agent claimed success, high role confidence, not ambiguous
    confident = [r for r in records if r.believed_success
                 and getattr(r, "role_confidence", 1.0) >= 0.6
                 and not getattr(r, "ambiguous", False)]
    sfc = float(np.mean([not r.success for r in confident])) if confident else 0.0
    uncertain = [r for r in records if r.believed_success
                 and (getattr(r, "ambiguous", False) or getattr(r, "role_confidence", 1.0) < 0.6)]
    correct_role = [r for r in records if getattr(r, "role_binding_correct", True)]
    p_s_given_role = float(np.mean([r.success for r in correct_role])) if correct_role else 0.0
    rba = float(np.mean([getattr(r, "role_binding_correct", True) for r in records]))
    plans = [x for r in records for x in r.plan_latencies_ms]
    steps = [x for r in records for x in r.step_latencies_ms]
    d2c = [r.disturbance_to_correction_steps for r in records
           if r.disturbance_to_correction_steps is not None]
    d2r = [r.disturbance_to_replan_steps for r in records
           if getattr(r, "disturbance_to_replan_steps", None) is not None]
    # -- resolution layer: autonomy vs safety --
    committed = [r for r in records if getattr(r, "committed", True)]
    auto = [r for r in committed if getattr(r, "clarifications", 0) == 0]
    asked = [r for r in records if getattr(r, "clarifications", 0) > 0]
    init_amb = [r for r in records if getattr(r, "initially_ambiguous", False)]
    selective_risk = float(np.mean([not r.success for r in committed])) if committed else 0.0
    autonomous_coverage = len(auto) / n
    abstention_rate = 1.0 - len(committed) / n
    clarification_rate = float(np.mean([getattr(r, "clarifications", 0) for r in records]))
    ambiguity_resolution_rate = (
        float(np.mean([getattr(r, "committed", True) and r.success for r in init_amb]))
        if init_amb else 0.0)
    post_res_rba = (float(np.mean([getattr(r, "role_binding_correct", True) for r in committed]))
                    if committed else 0.0)
    unnecessary_question_rate = (
        float(np.mean([not getattr(r, "initially_ambiguous", False) for r in asked]))
        if asked else 0.0)
    insp_per_ep = float(np.mean([getattr(r, "resolution_inspection_frames", 0) for r in records]))

    opps = [r for r in records if r.recovery_opportunity]
    rec = [r for r in opps if r.recovered]
    fails = [r for r in records if not r.success]
    fb = {}
    for r in fails:
        fb[r.failure_category] = fb.get(r.failure_category, 0) + 1
    cerr = {}
    for r in records:
        for k, v in r.context_error.items():
            cerr.setdefault(k, []).append(v)
    return dict(
        n=n, success_rate=float(np.mean(succ)), success_ci=bootstrap_ci(succ, seed=seed),
        first_attempt_rate=float(np.mean([r.first_attempt_success for r in records])),
        wrong_belief_rate=float(np.mean([r.wrong_belief for r in records])),
        silent_false_completion_rate=sfc,
        uncertain_completion_rate=len(uncertain) / n,
        p_success_given_correct_role=p_s_given_role,
        role_binding_accuracy=rba,
        recovery_opportunities=len(opps), recovered=len(rec),
        recovery_rate=(len(rec) / len(opps)) if opps else 0.0,
        autonomous_replans_total=sum(r.autonomous_replans for r in records),
        human_interventions_total=sum(r.human_interventions for r in records),
        plan_latency_ms={"p50": _pct(plans, 50), "p95": _pct(plans, 95),
                         "p99": _pct(plans, 99), "mean": float(np.mean(plans)) if plans else 0.0},
        step_latency_ms={"p50": _pct(steps, 50), "p95": _pct(steps, 95)},
        disturbance_to_correction_steps={"p50": _pct(d2c, 50), "n": len(d2c)},
        disturbance_to_replan_steps={"p50": _pct(d2r, 50), "n": len(d2r)},
        mean_completion_steps=float(np.mean([r.steps for r in records])),
        mean_completion_seconds=float(np.mean([r.sim_seconds for r in records])),
        collisions_per_ep=float(np.mean([r.collisions for r in records])),
        force_violations_per_ep=float(np.mean([r.force_violations for r in records])),
        irreversible_per_ep=float(np.mean([r.irreversible_failures for r in records])),
        safety_violations_per_ep=float(np.mean([r.safety_violations for r in records])),
        failure_breakdown=fb,
        context_error={k: float(np.mean(v)) for k, v in cerr.items()},
        selective_risk=selective_risk, autonomous_coverage=autonomous_coverage,
        abstention_rate=abstention_rate, clarification_rate=clarification_rate,
        ambiguity_resolution_rate=ambiguity_resolution_rate,
        post_resolution_role_accuracy=post_res_rba,
        unnecessary_question_rate=unnecessary_question_rate,
        inspection_frames_per_ep=insp_per_ep)


def aggregate(records: list[EpisodeRecord], seed=0) -> Report:
    base = _agg(records, seed=seed)
    by_split = {}
    splits = sorted(set(r.split for r in records))
    for sp in splits:
        rs = [r for r in records if r.split == sp]
        a = _agg(rs, seed=seed)
        by_split[sp] = {"success_rate": a["success_rate"], "ci": a["success_ci"], "n": a["n"]}
    return Report(by_split=by_split, **base)
