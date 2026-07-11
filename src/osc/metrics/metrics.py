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
    recovery_opportunities: int
    recovered: int
    recovery_rate: float
    autonomous_replans_total: int
    human_interventions_total: int
    plan_latency_ms: dict            # p50/p95/p99/mean over pooled calls
    step_latency_ms: dict            # sensor->action, pooled
    disturbance_to_correction_steps: dict
    mean_completion_steps: float
    mean_completion_seconds: float
    collisions_per_ep: float
    force_violations_per_ep: float
    irreversible_per_ep: float
    safety_violations_per_ep: float
    failure_breakdown: dict
    context_error: dict
    by_split: dict = field(default_factory=dict)

    def pretty(self) -> str:
        L = ["=" * 62,
             "  OSC v0.2 BENCHMARK REPORT  (belief-state, ground-truth scored)",
             "=" * 62,
             f"  episodes                      : {self.n}",
             f"  success rate                  : {self.success_rate:6.1%}  "
             f"CI95[{self.success_ci[0]:.2f},{self.success_ci[1]:.2f}]",
             f"  first-attempt success         : {self.first_attempt_rate:6.1%}",
             f"  wrong-belief rate (silent err): {self.wrong_belief_rate:6.1%}",
             f"  recovery opportunities        : {self.recovery_opportunities}"
             f"  -> recovered {self.recovered}  ({self.recovery_rate:.0%})",
             f"  autonomous replans (total)    : {self.autonomous_replans_total}",
             f"  human interventions (total)   : {self.human_interventions_total}",
             f"  plan latency ms  p50/p95/p99  : {self.plan_latency_ms['p50']:.1f} / "
             f"{self.plan_latency_ms['p95']:.1f} / {self.plan_latency_ms['p99']:.1f}",
             f"  sensor->action ms p50/p95     : {self.step_latency_ms['p50']:.2f} / "
             f"{self.step_latency_ms['p95']:.2f}",
             f"  disturbance->correction steps : p50={self.disturbance_to_correction_steps.get('p50',0):.0f}",
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
    plans = [x for r in records for x in r.plan_latencies_ms]
    steps = [x for r in records for x in r.step_latencies_ms]
    d2c = [r.disturbance_to_correction_steps for r in records
           if r.disturbance_to_correction_steps is not None]
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
        recovery_opportunities=len(opps), recovered=len(rec),
        recovery_rate=(len(rec) / len(opps)) if opps else 0.0,
        autonomous_replans_total=sum(r.autonomous_replans for r in records),
        human_interventions_total=sum(r.human_interventions for r in records),
        plan_latency_ms={"p50": _pct(plans, 50), "p95": _pct(plans, 95),
                         "p99": _pct(plans, 99), "mean": float(np.mean(plans)) if plans else 0.0},
        step_latency_ms={"p50": _pct(steps, 50), "p95": _pct(steps, 95)},
        disturbance_to_correction_steps={"p50": _pct(d2c, 50), "n": len(d2c)},
        mean_completion_steps=float(np.mean([r.steps for r in records])),
        mean_completion_seconds=float(np.mean([r.sim_seconds for r in records])),
        collisions_per_ep=float(np.mean([r.collisions for r in records])),
        force_violations_per_ep=float(np.mean([r.force_violations for r in records])),
        irreversible_per_ep=float(np.mean([r.irreversible_failures for r in records])),
        safety_violations_per_ep=float(np.mean([r.safety_violations for r in records])),
        failure_breakdown=fb,
        context_error={k: float(np.mean(v)) for k, v in cerr.items()})


def aggregate(records: list[EpisodeRecord], seed=0) -> Report:
    base = _agg(records, seed=seed)
    by_split = {}
    splits = sorted(set(r.split for r in records))
    for sp in splits:
        rs = [r for r in records if r.split == sp]
        a = _agg(rs, seed=seed)
        by_split[sp] = {"success_rate": a["success_rate"], "ci": a["success_ci"], "n": a["n"]}
    return Report(by_split=by_split, **base)
