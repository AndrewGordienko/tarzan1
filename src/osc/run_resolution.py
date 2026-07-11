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

from .benchmark.runner import run_benchmark
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
    args = ap.parse_args()
    base = ExecConfig()

    hdr = (f"{'config':26s} {'succ':>5} {'sel_risk':>8} {'auto_cov':>8} {'abstain':>7} "
           f"{'clar/ep':>7} {'amb_res':>7} {'postRBA':>7} {'insp/ep':>7}")
    print("=" * len(hdr)); print("  RESOLUTION ABLATION  (autonomy vs safety)"); print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, over in CONFIGS.items():
        cfg = replace(base, **over)
        r = aggregate(run_benchmark(seeds=range(args.seeds), cfg=cfg))
        print(f"{name:26s} {r.success_rate:5.2f} {r.selective_risk:8.3f} "
              f"{r.autonomous_coverage:8.3f} {r.abstention_rate:7.3f} {r.clarification_rate:7.2f} "
              f"{r.ambiguity_resolution_rate:7.3f} {r.post_resolution_role_accuracy:7.3f} "
              f"{r.inspection_frames_per_ep:7.1f}")
    print("-" * len(hdr))
    print("  sel_risk = P(fail | committed); auto_cov = P(committed & no question).")
    print("  Inspection cannot resolve genuine ties -> it abstains; clarification can.")


if __name__ == "__main__":
    main()
