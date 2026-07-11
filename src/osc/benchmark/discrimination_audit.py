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

import numpy as np

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


FEATURES = ["role_confidence", "role_margin", "assignment_margin", "role_entropy",
            "n_candidates", "track_uncertainty", "staleness", "top_cost", "second_cost"]


def feature_matrix(recs):
    X = [[float(getattr(r, f)) for f in FEATURES] for r in recs]
    return np.asarray(X, dtype=float)


# -- two simple, deterministic models (no sklearn; not a model search) --------
def _fit_logistic(X, y, iters=800, lr=0.2, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(Xs.shape[1]); b = 0.0
    yv = np.asarray(y, float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = p - yv
        w -= lr * (Xs.T @ g / len(yv) + l2 * w)
        b -= lr * g.mean()
    return ("logistic", w, b, mu, sd)


def _fit_gbstumps(X, y, rounds=40, lr=0.3):
    yv = np.asarray(y, float)
    base = float(np.clip(yv.mean(), 1e-3, 1 - 1e-3))
    F = np.full(len(yv), np.log(base / (1 - base)))
    stumps = []
    for _ in range(rounds):
        resid = yv - 1.0 / (1.0 + np.exp(-F))
        best = None
        for j in range(X.shape[1]):
            xs = X[:, j]
            for thr in np.unique(np.quantile(xs, [0.25, 0.5, 0.75])):
                left = xs <= thr
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                vl, vr = resid[left].mean(), resid[~left].mean()
                sse = ((resid - np.where(left, vl, vr)) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, j, thr, vl, vr)
        _, j, thr, vl, vr = best
        F += lr * np.where(X[:, j] <= thr, vl, vr)
        stumps.append((j, thr, vl, vr))
    return ("gbstumps", stumps, base, lr)


def _predict(model, X):
    if model[0] == "logistic":
        _, w, b, mu, sd = model
        return 1.0 / (1.0 + np.exp(-(((X - mu) / sd) @ w + b)))
    _, stumps, base, lr = model
    F = np.full(len(X), np.log(base / (1 - base)))
    for j, thr, vl, vr in stumps:
        F += lr * np.where(X[:, j] <= thr, vl, vr)
    return 1.0 / (1.0 + np.exp(-F))


def multivariate_probe(dev_seeds, held_seeds):
    """Fit LR + GB-stumps on DEV, evaluate ranking on HELD-OUT. Tests whether
    COMBINING weak features clears what single features cannot."""
    dev = run_benchmark(seeds=dev_seeds, cfg=ExecConfig())
    held = run_benchmark(seeds=held_seeds, cfg=ExecConfig())
    Xd, Xh = feature_matrix(dev), feature_matrix(held)
    out = {}
    for label, getter in (("binding", lambda r: bool(r.role_binding_correct)),
                          ("identifiable", lambda r: bool(r.identifiable))):
        yd = [getter(r) for r in dev]
        yh = [getter(r) for r in held]
        res = {}
        for name, fit in (("logistic", _fit_logistic), ("gbstumps", _fit_gbstumps)):
            model = fit(Xd, yd)
            ph = _predict(model, Xh)
            op = _operating_point(list(ph), [bool(r.identifiable) for r in held]) \
                if label == "identifiable" else None
            res[name] = dict(auroc=_auroc(list(ph), yh), op=op)
        out[label] = res
    # feature correlations on dev
    C = np.corrcoef(Xd.T)
    return dict(models=out, corr=C, held_n=len(held))


def clarification_decomposition(seeds=range(80)):
    """Split ambiguity_resolution into resolution vs downstream execution, on the
    GENUINELY-AMBIGUOUS scenes that were clarified. The oracle answer + immediate
    override make response/immediate-binding ~correct by construction, so the
    measurable chain is: persistent-binding (survives replans) -> task success."""
    cfg = ExecConfig(resolution=True, allow_inspection=False, allow_clarification=True)
    recs = run_benchmark(seeds=seeds, cfg=cfg)
    amb_clar = [r for r in recs if (not r.identifiable) and r.clarifications > 0]
    n = len(amb_clar)
    rbc = [r for r in amb_clar if r.role_binding_correct]
    breakpoints = dict(
        binding_not_persistent=sum(1 for r in amb_clar if not r.role_binding_correct),
        correct_binding_control_fail=sum(1 for r in rbc if not r.success
                                         and r.failure_category == "control"),
        correct_binding_verifier_reject=sum(1 for r in rbc if not r.success
                                            and r.failure_category == "verification_false_positive"),
        correct_binding_other_fail=sum(1 for r in rbc if not r.success
                                       and r.failure_category not in
                                       ("control", "verification_false_positive")),
        fully_resolved=sum(1 for r in rbc if r.success))
    return dict(
        n=n,
        persistent_binding=(len(rbc) / n) if n else 0.0,
        success_given_binding=(sum(r.success for r in rbc) / len(rbc)) if rbc else 0.0,
        overall_success=(sum(r.success for r in amb_clar) / n) if n else 0.0,
        breakpoints=breakpoints)


def audit(seeds=range(80)):
    recs = run_benchmark(seeds=seeds, cfg=ExecConfig())      # resolution off: raw confidence
    conf = [r.role_confidence for r in recs]
    bind = [bool(r.role_binding_correct) for r in recs]
    ident = [bool(r.identifiable) for r in recs]
    succ = [bool(r.success) for r in recs]
    return dict(recs=recs, conf=conf, bind=bind, ident=ident, succ=succ,
                features=feature_table(recs))
