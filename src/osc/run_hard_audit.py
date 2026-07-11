"""Run the paired stratified perception re-audit and save machine-readable data."""
from __future__ import annotations

import argparse
import json

from .benchmark.hard_perception import oracle_viewpoint_ladder, paired_hard_audit
from .benchmark.runner import _json_default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--out", default="hard_perception_audit.json")
    ap.add_argument("--no-ladder", action="store_true")
    args = ap.parse_args()
    report = paired_hard_audit(seeds=range(args.seeds))
    if not args.no_ladder:
        report["oracle_viewpoint_ladder"] = oracle_viewpoint_ladder(seeds=range(args.seeds))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    for name, data in report["reports"].items():
        r = data["overall"]
        op = r["max_identifiable_coverage_below_5pct_ambiguous"]
        print(f"{name:24s} bind AUROC {r['binding_correct_auroc']:.3f}  "
              f"ident AUROC {r['identifiability_auroc']:.3f}  "
              f"autonomous-ident {r['correct_autonomous_commit_identifiable']:.3f}  "
              f"ambiguous-commit {r['ambiguous_commit_rate']:.3f}  "
              f"success {r['end_to_end_success']:.3f}  op={op}")
        m = data["identifiability_metric_validation"]
        print(f"  ident metric: pooled={m['pooled_auroc']:.3f} random={m['random_score_auroc']:.3f} "
              f"gt={m['ground_truth_identifiability_auroc']:.3f} negated={m['negated_score_auroc']:.3f} "
              f"macro-stratum={m['macro_stratum_auroc']:.3f}")
    print("flip analysis:", {k: v for k, v in report["flip_analysis"].items()
                             if k != "diagnostics"})


if __name__ == "__main__":
    main()
