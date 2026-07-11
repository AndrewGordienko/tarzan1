"""Resolution-layer ablation: the autonomy-vs-safety curve.

Compares the ResolutionPolicy configurations the v0.4 gate is about. The point is
NOT raw success -- refusing more work always looks "safer". The core trade-off is
AUTONOMOUS COVERAGE (completed without asking) vs SELECTIVE RISK (wrong among the
episodes the robot committed to), separating what active INSPECTION can fix
(poor sensing) from what only CLARIFICATION can (fundamental ambiguity).
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from .benchmark.runner import run_benchmark, run_workflows
from .execution.loop import ExecConfig
from .metrics.metrics import aggregate

CONFIGS = {
    "none (guess & commit)": dict(resolution=False),
    "passive/inspection-only": dict(resolution=True, allow_inspection=True, allow_clarification=False),
    "clarification-only": dict(resolution=True, allow_inspection=False, allow_clarification=True),
    "combined": dict(resolution=True, allow_inspection=True, allow_clarification=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--workflows", type=int, default=8)
    ap.add_argument("--orders", type=int, default=20)
    args = ap.parse_args()
    base = ExecConfig()

    hdr = (f"{'config':26s} {'succ':>5} {'anyRisk':>7} {'silent':>7} {'roleErr':>7} "
           f"{'covIDENT':>8} {'covAMB':>7} {'abstain':>7} {'clar/ep':>7} {'ambRes<=1Q':>10}")
    print("=" * len(hdr))
    print("  RESOLUTION ABLATION  (autonomy vs safety, decomposed)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for name, over in CONFIGS.items():
        r = aggregate(run_benchmark(seeds=range(args.seeds), cfg=replace(base, **over)))
        print(f"{name:26s} {r.success_rate:5.2f} {r.selective_risk:7.3f} "
              f"{r.risk_silent_committed:7.3f} {r.risk_role_wrong:7.3f} "
              f"{r.auto_cov_identifiable:8.3f} {r.auto_cov_ambiguous:7.3f} {r.abstention_rate:7.3f} "
              f"{r.clarification_rate:7.2f} "
              f"{r.ambiguity_resolution_rate:.2f}[{r.ci_ambiguity_resolution[0]:.2f},{r.ci_ambiguity_resolution[1]:.2f}]")
    print("-" * len(hdr))
    print("  anyRisk/silent/roleErr = P(.|committed);  covIDENT = auto-coverage on identifiable")
    print("  scenes (gate >0.80);  covAMB should be ~0;  ambRes = P(success,<=1 question|ambiguous).")

    # -- per-WORKFLOW clarification cost (one demo + N production orders) --
    print("\n  clarifications per WORKFLOW  (combined policy, "
          f"{args.workflows} workflows x {args.orders} orders):")
    w = run_workflows(n_workflows=args.workflows, orders_per_workflow=args.orders,
                      cfg=replace(base, resolution=True, allow_inspection=True, allow_clarification=True))
    print(f"    clarifications / workflow            : {w['clarifications_per_workflow']:.2f}")
    print(f"    clarifications / production episode  : {w['clarifications_per_production_ep']:.3f}")
    print(f"    repeated-question rate (post-setup)  : {w['repeated_question_rate']:.3f}")
    print(f"    production role accuracy (transfer)  : {w['production_role_accuracy']:.3f}")
    print(f"    production success (transfer)        : {w['production_success']:.3f}")


if __name__ == "__main__":
    main()
