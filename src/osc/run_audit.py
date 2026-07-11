"""Discrimination + risk-coverage audit of the role-binding confidence score.

Answers: can ANY threshold on the current confidence separate identifiable from
ambiguous scenes well enough to hit the v0.4 gate? If AUROC is near 0.5, the
answer is no and the evidence representation -- not the threshold -- must change.
"""
from __future__ import annotations

import argparse

from .benchmark import discrimination_audit as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=80)
    args = ap.parse_args()
    d = A.audit(seeds=range(args.seeds))
    conf, bind, ident, succ, recs = d["conf"], d["bind"], d["ident"], d["succ"], d["recs"]
    n = len(recs)

    print("=" * 64)
    print(f"  DISCRIMINATION AUDIT  (confidence = min per-role marginal)   n={n}")
    print("=" * 64)
    print(f"  base rates: binding-correct {sum(bind)/n:.2f}  identifiable {sum(ident)/n:.2f}")
    print("\n  RANKING power (can confidence order the classes?)")
    print(f"    AUROC binding_correct   : {A._auroc(conf, bind):.3f}   "
          f"AUPRC {A._auprc(conf, bind):.3f}")
    print(f"    AUROC identifiable      : {A._auroc(conf, ident):.3f}   "
          f"AUPRC {A._auprc(conf, ident):.3f}")
    print("    (0.5 = no signal; scalar calibration cannot raise these)")

    print("\n  per-FEATURE ranking power (which agent-visible signal separates best?)")
    print(f"    {'feature':26s} {'AUROC-bind':>10} {'AUROC-ident':>12}")
    best_feat = (0.0, None)
    for name, xs in d["features"].items():
        ab, ai = A._auroc(xs, bind), A._auroc(xs, ident)
        print(f"    {name:26s} {ab:10.3f} {ai:12.3f}")
        best_feat = max(best_feat, (ab, name))
    print(f"    best single feature for binding: {best_feat[1]} (AUROC {best_feat[0]:.3f})")

    print("\n  per split  (AUROC binding / AUROC identifiable):")
    for sp in sorted(set(r.split for r in recs)):
        idx = [i for i, r in enumerate(recs) if r.split == sp]
        c = [conf[i] for i in idx]; b = [bind[i] for i in idx]; ii = [ident[i] for i in idx]
        print(f"    {sp:24s} {A._auroc(c, b):.3f} / {A._auroc(c, ii):.3f}   n={len(idx)}")

    print("\n  best coverage at selective-risk target (binding error), any threshold:")
    for t in (0.05, 0.10):
        print(f"    risk<={t:.2f} -> coverage {A._best_coverage_at_risk(conf, bind, t):.3f}")

    print("\n  reliability of confidence as P(binding correct):")
    ece, brier, rows = A._reliability(conf, bind)
    print(f"    ECE {ece:.3f}   Brier {brier:.3f}")
    for lo, hi, k, cf, ac in rows:
        print(f"      [{lo:.1f},{hi:.1f})  n={k:3d}  conf={cf:.2f}  acc={ac:.2f}")

    print("\n  *** DECISIVE OPERATING POINT ***  (best over ALL agent-visible features)")
    best = (0.0, None, None, None)
    for name, xs in d["features"].items():
        op = A._operating_point(xs, ident, amb_budget=0.05)
        if op:
            cov, tau, amb = op
            print(f"    {name:26s} coverage={cov:.3f} @tau={tau:.3f} (amb-commit={amb:.3f})")
            best = max(best, (cov, name, tau, amb))
    print(f"    -> best achievable identifiable-coverage @ <5% ambiguous-commit: "
          f"{best[0]:.3f} via {best[1]}")
    if best[0] >= 0.80:
        print("    => VIABLE operating point exists on the current evidence.")
    else:
        print("    => NO VIABLE OPERATING POINT (gate wants >=0.80) on ANY current feature.")
        print("       Binding-correctness is ~unrankable (AUROC~0.6); identifiability tops")
        print("       out ~0.80 -- too low for 80% coverage at 5% FPR. Conclusion: do NOT")
        print("       tune/calibrate a score over the CURRENT evidence. The agent must")
        print("       GATHER more evidence (real active inspection / interaction in the")
        print("       perception path) -- i.e. change the evidence, not the threshold.")


if __name__ == "__main__":
    main()
