"""Discrimination + risk-coverage audit for the role-binding confidence score.

Run BEFORE touching the policy. If the confidence score cannot RANK correct vs
wrong bindings (and identifiable vs ambiguous scenes), then no threshold and no
scalar calibration (temperature / isotonic) can fix it -- the evidence
representation must change. Scalar calibration only re-maps probabilities; it
cannot improve ranking.

The decisive question:
  Does ANY single threshold on the current confidence achieve >80% autonomous
  coverage on identifiable scenes while committing on <5% of fundamentally
  ambiguous scenes?

Uses only agent-visible confidence + the scorer's objective labels. No sklearn.
"""
from __future__ import annotations

from .runner import run_benchmark
from ..execution.loop import ExecConfig


def _auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    u = 0.0
    for p in pos:
        for q in neg:
            u += 1.0 if p > q else (0.5 if p == q else 0.0)
    return u / (len(pos) * len(neg))


def _auprc(scores, labels):
    P = sum(labels)
    if P == 0:
        return float("nan")
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for s, l in sorted(zip(scores, labels), key=lambda x: -x[0]):
        if l:
            tp += 1
        else:
            fp += 1
        recall = tp / P
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def _risk_coverage(scores, correct):
    """Sort by confidence desc; report (coverage, selective-risk) as we admit more."""
    order = sorted(zip(scores, correct), key=lambda x: -x[0])
    n = len(order)
    out, errs = [], 0
    for k, (_, c) in enumerate(order, start=1):
        errs += 0 if c else 1
        out.append((k / n, errs / k))
    return out


def _best_coverage_at_risk(scores, correct, target):
    """Max coverage achievable by ANY single threshold with selective risk<=target."""
    best = 0.0
    for tau in sorted(set(scores)):
        committed = [(s >= tau) for s in scores]
        m = sum(committed)
        if m == 0:
            continue
        risk = sum(1 for s, c in zip(scores, correct) if s >= tau and not c) / m
        if risk <= target:
            best = max(best, m / len(scores))
    return best


def _operating_point(scores, identifiable, amb_budget=0.05):
    """Best identifiable-coverage under any threshold s.t. commit-rate on
    fundamentally-ambiguous scenes stays < amb_budget."""
    ident = [s for s, i in zip(scores, identifiable) if i]
    amb = [s for s, i in zip(scores, identifiable) if not i]
    if not ident or not amb:
        return None
    best = (0.0, None, 0.0)
    for tau in sorted(set(scores)):
        amb_commit = sum(1 for s in amb if s >= tau) / len(amb)
        if amb_commit < amb_budget:
            cov = sum(1 for s in ident if s >= tau) / len(ident)
            if cov > best[0]:
                best = (cov, tau, amb_commit)
    return best


def _reliability(scores, correct, bins=10):
    ece = 0.0
    n = len(scores)
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if (lo <= s < hi or (b == bins - 1 and s == 1.0))]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
        rows.append((lo, hi, len(idx), conf, acc))
    brier = sum((s - (1.0 if c else 0.0)) ** 2 for s, c in zip(scores, correct)) / n
    return ece, brier, rows


def feature_table(recs):
    """Agent-visible features that a resolvability model could use. Signs chosen so
    HIGHER = more likely correct/identifiable (entropy & near-ties are negated)."""
    return {
        "confidence(min-marginal)": [r.role_confidence for r in recs],
        "role_margin(top1-top2)": [r.role_margin for r in recs],
        "assignment_margin": [r.assignment_margin for r in recs],
        "neg_entropy": [-r.role_entropy for r in recs],
    }


def audit(seeds=range(80)):
    recs = run_benchmark(seeds=seeds, cfg=ExecConfig())      # resolution off: raw confidence
    conf = [r.role_confidence for r in recs]
    bind = [bool(r.role_binding_correct) for r in recs]
    ident = [bool(r.identifiable) for r in recs]
    succ = [bool(r.success) for r in recs]
    return dict(recs=recs, conf=conf, bind=bind, ident=ident, succ=succ,
                features=feature_table(recs))
