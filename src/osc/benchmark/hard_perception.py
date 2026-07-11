"""Paired, stratified perception benchmark for the resolution loop.

Every configuration is run on the same task/split/seed blocks.  The oracle
perception row retains ordinary role correspondence; oracle role binding is a
separate upper bound, preventing those two gains from being conflated.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .discrimination_audit import _auprc, _auroc, _operating_point, _risk_coverage
from .runner import Split, run_benchmark
from ..execution.loop import ExecConfig
from ..perception.detections import CorruptionSpec
from ..sim.randomize import RandomizationSpec


def _distribution(xs):
    a = np.asarray(xs, dtype=float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else float("nan"),
                std=float(a.std()) if len(a) else float("nan"),
                q25=float(np.quantile(a, .25)) if len(a) else float("nan"),
                q50=float(np.quantile(a, .50)) if len(a) else float("nan"),
                q75=float(np.quantile(a, .75)) if len(a) else float("nan"))


def validate_identifiability_metric(records) -> dict:
    """Validate direction, label plumbing, and pooled-vs-stratified effects.

    This is intentionally reported beside every hard-audit row.  A low pooled
    AUROC is never promoted to a policy gate until random/GT/negation controls
    and per-stratum direction have been inspected.
    """
    labels = [bool(r.identifiable) for r in records]
    scores = [float(r.role_confidence) for r in records]
    rng = np.random.default_rng(917)
    random_scores = list(rng.random(len(records)))
    by_stratum = {}
    for split in sorted({r.split for r in records}):
        idx = [i for i, r in enumerate(records) if r.split == split]
        ss, yy = [scores[i] for i in idx], [labels[i] for i in idx]
        by_stratum[split] = dict(n=len(idx), label_balance=sum(yy) / max(1, len(yy)),
                                 auroc=_auroc(ss, yy), negated_auroc=_auroc([-s for s in ss], yy),
                                 identifiable_scores=_distribution([s for s, y in zip(ss, yy) if y]),
                                 ambiguous_scores=_distribution([s for s, y in zip(ss, yy) if not y]))
    stratum_aurocs = [d["auroc"] for d in by_stratum.values() if not np.isnan(d["auroc"])]
    return dict(
        pooled_auroc=_auroc(scores, labels),
        random_score_auroc=_auroc(random_scores, labels),
        ground_truth_identifiability_auroc=_auroc([float(y) for y in labels], labels),
        negated_score_auroc=_auroc([-s for s in scores], labels),
        label_balance=dict(identifiable=sum(labels), ambiguous=len(labels) - sum(labels), n=len(labels)),
        identifiable_score_distribution=_distribution([s for s, y in zip(scores, labels) if y]),
        ambiguous_score_distribution=_distribution([s for s, y in zip(scores, labels) if not y]),
        macro_stratum_auroc=float(np.mean(stratum_aurocs)) if stratum_aurocs else float("nan"),
        by_stratum=by_stratum,
    )


def hard_perception_splits() -> list[Split]:
    """Three known categories: noise (fusion), hidden geometry (viewpoint), and
    fundamental ambiguity (clarify/abstain).  All include shuffled detections,
    occlusion, close distractors, and track loss/reacquisition pressure."""
    corr = CorruptionSpec(pos_noise=0.006, size_noise=0.004, occlusion_prob=0.16,
                          drop_prob=0.07, delay_frames=0, false_contact_prob=0.03,
                          identity_swap_prob=0.10)
    common = dict(camera_model=True, n_distractors=3, distractor_min_separation=0.055,
                  role_size_jitter=0.05)
    return [
        Split("hard_noisy_visible", RandomizationSpec(**common, role_feature_mode="native"), corr),
        Split("hard_hidden_default_view",
              RandomizationSpec(**common, role_feature_mode="height_only"), corr),
        Split("hard_fundamental_ambiguity",
              RandomizationSpec(**common, role_feature_mode="identical"), corr),
    ]


def paired_configs() -> dict[str, tuple[ExecConfig, bool, bool]]:
    """name -> (executor config, oracle perception, oracle role binding)."""
    # The production policy's legacy 0.60 threshold accepts several deliberately
    # weak 0.62--0.68 hard-split assignments before it can inspect them.  The
    # paired audit uses 0.75 consistently to make the stated "unresolved" branch
    # a genuine test of evidence gathering rather than a threshold ablation.
    base = dict(resolution=True, allow_clarification=True, commit_threshold=0.75)
    return {
        "single_frame": (ExecConfig(**base, allow_inspection=False, allow_viewpoint=False), False, False),
        "passive_fusion": (ExecConfig(**base, allow_inspection=True, allow_viewpoint=False), False, False),
        "passive_plus_viewpoint": (ExecConfig(**base, allow_inspection=True, allow_viewpoint=True), False, False),
        # Oracle perception retains normal RoleBelief/correspondence.
        "oracle_perception": (ExecConfig(**base, allow_inspection=True, allow_viewpoint=True), True, False),
        "oracle_role_binding": (ExecConfig(**base, allow_inspection=True, allow_viewpoint=True), False, True),
    }


def _summary(records) -> dict:
    scores = [float(r.role_confidence) for r in records]
    binding = [bool(r.role_binding_correct) for r in records]
    identifiable = [bool(r.identifiable) for r in records]
    committed = [bool(r.committed) for r in records]
    autonomous = [c and r.clarifications == 0 for c, r in zip(committed, records)]
    n = max(1, len(records))
    committed_n = sum(committed)
    return dict(
        n=len(records),
        binding_correct_auroc=_auroc(scores, binding),
        binding_correct_auprc=_auprc(scores, binding),
        identifiability_auroc=_auroc(scores, identifiable),
        identifiability_auprc=_auprc(scores, identifiable),
        risk_coverage=_risk_coverage(scores, binding),
        max_identifiable_coverage_below_5pct_ambiguous=_operating_point(scores, identifiable, 0.05),
        correct_autonomous_commit_identifiable=(sum(c and b and i for c, b, i in zip(autonomous, binding, identifiable)) /
                                                max(1, sum(identifiable))),
        # Clarified episodes are supported by external information; this is the
        # explicitly *autonomous*, unsupported-commit rate used by the <5% gate.
        ambiguous_commit_rate=(sum(c and not i for c, i in zip(autonomous, identifiable)) /
                               max(1, sum(not i for i in identifiable))),
        silent_role_binding_error_among_committed=(sum(c and not b for c, b in zip(committed, binding)) /
                                                    max(1, committed_n)),
        association_contested_rate=sum(bool(r.association_contested) for r in records) / n,
        correct_binding_after_viewpoint=(sum(r.role_binding_correct for r in records if r.viewpoint_actions) /
                                         max(1, sum(bool(r.viewpoint_actions) for r in records))),
        unnecessary_viewpoint_rate=(sum(bool(r.viewpoint_actions) and not r.identifiable for r in records) /
                                    max(1, sum(bool(r.viewpoint_actions) for r in records))),
        mean_views=float(np.mean([len(r.viewpoints) for r in records])),
        mean_view_actions=float(np.mean([r.viewpoint_actions for r in records])),
        mean_additional_frames=float(np.mean([r.resolution_inspection_frames + r.viewpoint_frames for r in records])),
        mean_steps=float(np.mean([r.steps for r in records])),
        mean_latency_ms=float(np.mean([sum(r.plan_latencies_ms) + sum(r.step_latencies_ms) for r in records])),
        mean_clarifications=float(np.mean([r.clarifications for r in records])),
        end_to_end_success=float(np.mean([r.success for r in records])),
    )


def paired_hard_audit(seeds=range(20), tasks=None) -> dict:
    """Run and return the complete paired re-audit, including task/category
    strata and deltas from the single-frame baseline."""
    splits = hard_perception_splits()
    raw, reports = {}, {}
    for name, (cfg, oracle_perception, oracle_binding) in paired_configs().items():
        recs = run_benchmark(tasks=tasks, splits=splits, seeds=seeds, cfg=cfg,
                             privileged=oracle_perception,
                             oracle_role_binding=oracle_binding)
        raw[name] = recs
        overall = _summary(recs)
        strata = {}
        for task in sorted({r.task for r in recs}):
            for category in sorted({r.split for r in recs}):
                rs = [r for r in recs if r.task == task and r.split == category]
                strata[f"{task}/{category}"] = _summary(rs)
        reports[name] = dict(overall=overall, by_task_category=strata)
        reports[name]["identifiability_metric_validation"] = validate_identifiability_metric(recs)
    baseline = reports["single_frame"]["overall"]
    for name, report in reports.items():
        report["delta_vs_single_frame"] = {
            key: report["overall"][key] - baseline[key]
            for key in ("end_to_end_success", "mean_steps", "mean_latency_ms", "mean_clarifications",
                        "silent_role_binding_error_among_committed", "association_contested_rate")
        }
        report["clarifications_avoided_vs_single_frame"] = -report["delta_vs_single_frame"]["mean_clarifications"]
    flips = paired_flip_analysis(raw["passive_fusion"], raw["passive_plus_viewpoint"])
    return dict(reports=reports, flip_analysis=flips,
                records={k: [asdict(r) for r in v] for k, v in raw.items()})


def paired_flip_analysis(passive_records, viewpoint_records) -> dict:
    """Paired seed-level explanation for viewpoint regressions."""
    key = lambda r: (r.task, r.split, r.seed)
    passive = {key(r): r for r in passive_records}
    viewed = {key(r): r for r in viewpoint_records}
    pairs = [(passive[k], viewed[k]) for k in sorted(passive.keys() & viewed.keys())]
    diags = [d for _, v in pairs for d in v.viewpoint_diagnostics]
    return dict(
        n_pairs=len(pairs),
        passive_right_viewpoint_wrong=sum(p.role_binding_correct and not v.role_binding_correct for p, v in pairs),
        viewpoint_repairs=sum(not p.role_binding_correct and v.role_binding_correct for p, v in pairs),
        viewpoint_introduced_timeouts=sum(not p.timeout and v.timeout for p, v in pairs),
        viewpoint_added_steps=float(np.mean([v.steps - p.steps for p, v in pairs])) if pairs else 0.0,
        viewed_episodes=sum(bool(v.viewpoint_diagnostics) for _, v in pairs),
        correct_to_wrong_during_view=sum(d["binding_before_correct"] and not d["binding_after_correct"] for d in diags),
        wrong_to_correct_during_view=sum(not d["binding_before_correct"] and d["binding_after_correct"] for d in diags),
        track_churn_during_view=sum(d["track_identity_changed"] for d in diags),
        nis_rejections_during_view=sum(d["nis_rejections"] for d in diags),
        covariance_only_assignment_changes=sum(d["assignment_changed_from_covariance_only"] for d in diags),
        expected_information_gain=float(np.mean([d["expected_information_gain"] for d in diags])) if diags else 0.0,
        realized_information_gain=float(np.mean([d["realized_information_gain"] for d in diags])) if diags else 0.0,
        discriminating_feature_revealed_rate=(sum(d["selected_view_revealed_discriminating_feature"] for d in diags) /
                                              max(1, len(diags))),
        diagnostics=diags,
    )


def oracle_viewpoint_ladder(seeds=range(20), tasks=None) -> dict:
    """Evidence/estimation/selection/matcher ladder on paired hidden-view seeds.

    The first two rows are decisive oracle-track controls; the remaining rows
    execute the normal benchmark with a forced calibrated best view (FRONT for
    the height-only split) or the learned selector.  Oracle correspondence is
    intentionally separate from oracle perception.
    """
    from .viewpoint_controls import geometry_height_control, side_marker_control
    hidden = [hard_perception_splits()[1]]
    base = dict(resolution=True, allow_inspection=True, allow_clarification=True,
                commit_threshold=.99, allow_viewpoint=True)
    rows = {}
    for name, oracle_binding, cfg in (
        ("estimated_tracks_oracle_best_view", False,
         ExecConfig(**base, viewpoint_candidates=("front",))),
        ("estimated_tracks_selected_view", False, ExecConfig(**base)),
        ("oracle_correspondence_after_selected_view", True, ExecConfig(**base)),
        ("full_system", False, ExecConfig(**base)),
    ):
        recs = run_benchmark(tasks=tasks, splits=hidden, seeds=seeds, cfg=cfg,
                             oracle_role_binding=oracle_binding)
        rows[name] = _summary(recs)
    controls = {"geometry_height": geometry_height_control(), "side_marker": side_marker_control()}
    return dict(
        oracle_tracks_top_only={k: v["oracle_tracks_top_only"] for k, v in controls.items()},
        oracle_tracks_oracle_best_view={k: v["oracle_tracks_correct_view"] for k, v in controls.items()},
        oracle_tracks_wrong_view={k: v["oracle_tracks_wrong_view"] for k, v in controls.items()},
        benchmark_rows=rows,
    )
