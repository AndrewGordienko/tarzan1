"""Exact-assignment isolation ladder and posterior diagnostics.

This module is deliberately separate from commitment policy evaluation.  It
answers whether the correct complete assignment is present/ranked before any
threshold or clarification decision is changed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from .hard_perception import hard_perception_splits
from .runner import run_benchmark
from ..execution.loop import ExecConfig
from ..perception.detections import CorruptionSpec
from ..sim.randomize import RandomizationSpec


def _wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def _summary(records):
    ds = [r.assignment_diagnostics for r in records if r.assignment_diagnostics]
    ranks = [d.get("gt_assignment_rank") for d in ds]
    present = [d.get("gt_assignment_present", False) for d in ds]
    top = [d.get("top_assignment_correct", False) for d in ds]
    committed = [r.committed for r in records]
    classes = Counter(r.assignment_failure_class for r in records)
    observed = Counter(tuple(sorted((k, tuple(v)) for k, v in d.get("observed_dimensions", {}).items()))
                      for d in ds)
    # Excluded GT assignments count as not-first for the episode-level rate;
    # otherwise top-K truncation would look artificially better by dropping hard
    # episodes from the denominator.
    rank_first = [x == 1 for x in ranks]
    silent = [r.committed and not r.role_binding_correct for r in records]
    return dict(
        n=len(records), diagnostics_n=len(ds),
        gt_assignment_present_rate=float(np.mean(present)) if present else float("nan"),
        gt_rank_first_rate=float(np.mean(rank_first)) if rank_first else float("nan"),
        gt_rank_first_ci=_wilson(sum(rank_first), len(rank_first)),
        gt_rank_mean=float(np.mean([x for x in ranks if x is not None])) if any(x is not None for x in ranks) else float("nan"),
        gt_rank_distribution=dict(Counter(str(x) for x in ranks)),
        gt_assignment_top_k_excluded=sum(not x for x in present),
        top_assignment_correct_rate=float(np.mean(top)) if top else float("nan"),
        top_assignment_correct_ci=_wilson(sum(top), len(top)),
        committed_rate=float(np.mean(committed)) if committed else float("nan"),
        silent_committed_binding_error=float(np.mean([r.committed and not r.role_binding_correct for r in records])) if records else float("nan"),
        silent_committed_binding_error_ci=_wilson(sum(silent), len(silent)),
        null_mass_mean=float(np.mean([d.get("null_mass", 0.0) for d in ds])) if ds else float("nan"),
        outside_top_k_mass_mean=float(np.mean([d.get("posterior_mass_outside_top_k", 0.0) for d in ds])) if ds else float("nan"),
        log_likelihood_gap_mean=float(np.mean([d.get("log_likelihood_gap", 0.0) for d in ds])) if ds else float("nan"),
        per_role_marginals=[d.get("per_role_marginals", {}) for d in ds],
        observed_dimensions=dict((str(k), v) for k, v in observed.items()),
        failure_classes=dict(classes),
        association_contested_rate=float(np.mean([r.association_contested for r in records])) if records else float("nan"),
        nis_rejection_rate=float(np.mean([sum(not e.get("accepted", True) for e in d.get("nis_events", []))
                                          for r in records for d in r.viewpoint_diagnostics])) if any(r.viewpoint_diagnostics for r in records) else 0.0,
    )


def run_correspondence_isolation(seeds=range(100), tasks=None):
    """Run paired same-seed conditions plus null/normalization/floor/constraint ablations."""
    hidden = hard_perception_splits()[1]
    # Stable-identity controls remove domain-jitter from the role objects while
    # retaining the same two-role geometry; this isolates correspondence from
    # the random-instance identifiability label by construction.
    stable_rand = replace(hidden.rand, role_size_jitter=0.0, n_distractors=0)
    noiseless = replace(hidden, name="oracle_noiseless_stable", rand=stable_rand,
                        corr=CorruptionSpec(enabled=False))
    noisy = replace(hidden, name="oracle_noisy_stable", rand=stable_rand)
    base = dict(resolution=True, allow_inspection=True, allow_clarification=True,
                commit_threshold=.75)
    configs = {
        "noiseless_observations_stable_oracle_identities": (noiseless, ExecConfig(**base), True),
        "noisy_observations_stable_oracle_identities": (noisy, ExecConfig(**base), True),
        "estimated_tracks_exact_assignment": (hidden, ExecConfig(**base, assignment_top_k=None), False),
        "estimated_tracks_production_approximation": (hidden, ExecConfig(**base, assignment_top_k=2), False),
        "full_loop": (hidden, ExecConfig(**base), False),
        "ablate_nulls": (hidden, ExecConfig(**base, assignment_allow_null=False), False),
        "ablate_dimension_normalization": (hidden, ExecConfig(**base, assignment_normalize_dimensions=True), False),
        "ablate_covariance_floor": (hidden, ExecConfig(**base, assignment_covariance_floor=0.0), False),
        "ablate_model_error_floor": (hidden, ExecConfig(**base, assignment_model_error_var=0.0), False),
        "ablate_clarification_constraints": (hidden, ExecConfig(**base, honor_clarification_constraints=False), False),
    }
    out = {}
    episodes = {}
    for name, (split, cfg, privileged) in configs.items():
        records = run_benchmark(tasks=tasks, splits=[split], seeds=seeds, cfg=cfg,
                                privileged=privileged)
        out[name] = _summary(records)
        episodes[name] = [dict(task=r.task, split=r.split, seed=r.seed,
                               committed=r.committed, clarifications=r.clarifications,
                               role_binding_correct=r.role_binding_correct,
                               success=r.success,
                               assignment_failure_class=r.assignment_failure_class,
                               assignment_diagnostics=r.assignment_diagnostics,
                               association_contested=r.association_contested,
                               nis_rejections=sum(d.get("nis_rejections", 0)
                                                  for d in r.viewpoint_diagnostics),
                               track_churn=sum(bool(d.get("track_identity_changed", False))
                                               for d in r.viewpoint_diagnostics))
                         for r in records]
    return dict(summary=out, episodes=episodes)
