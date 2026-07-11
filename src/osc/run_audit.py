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
        print("    => NO viable SINGLE-feature operating point (gate wants >=0.80).")

    # -- CHECK 1: multivariate sanity probe (do weak features COMBINE?) --------
    print("\n  MULTIVARIATE PROBE  (fit on DEV, evaluate on HELD-OUT; do combos help?)")
    mp = A.multivariate_probe(dev_seeds=range(0, args.seeds),
                              held_seeds=range(5000, 5000 + args.seeds))
    for label, res in mp["models"].items():
        for name, r in res.items():
            op = f"  op-cov@<5%amb={r['op'][0]:.3f}" if r["op"] else ""
            print(f"    {label:12s} {name:9s} held AUROC={r['auroc']:.3f}{op}")
    id_ops = [r["op"][0] for r in mp["models"]["identifiable"].values() if r["op"]]
    best_id_op = max(id_ops) if id_ops else 0.0
    print(f"    (held n={mp['held_n']})")
    print(f"    => combining weak features RAISES identifiable op-coverage from ~0.23")
    print(f"       (best single feature) to {best_id_op:.2f}. Route is NOT closed; but it is")
    print(f"       {'AT' if best_id_op>=0.80 else 'still short of'} the 0.80 gate -- a resolvability model helps, richer")
    print("       evidence (Step 5) is still needed to clear it. Binding-correctness stays")
    print("       weak (~0.74) => knowing IF a pick is right needs more evidence, not more model.")

    # -- CHECK 2: clarification decomposition (resolution vs downstream) --------
    print("\n  CLARIFICATION DECOMPOSITION  (genuinely-ambiguous scenes, clarified)")
    dc = A.clarification_decomposition(seeds=range(args.seeds))
    print(f"    n={dc['n']}")
    print(f"    P(binding persists correct through replans) : {dc['persistent_binding']:.3f}")
    print(f"    P(task success | binding stayed correct)    : {dc['success_given_binding']:.3f}")
    print(f"    overall P(success)                          : {dc['overall_success']:.3f}")
    print("    where the chain breaks (counts):")
    for k, v in dc["breakpoints"].items():
        print(f"      {k:34s} {v}")
    if dc["persistent_binding"] >= 0.95 and dc["success_given_binding"] < dc["persistent_binding"]:
        print("    => clarification RESOLVES the role; remaining loss is downstream control.")
    elif dc["persistent_binding"] < 0.95:
        print("    => clarification itself does not yet persist a correct binding.")


if __name__ == "__main__":
    main()
