"""Ablation harness: turn one component off at a time and measure the drop.

Each row shares the same tasks/splits/seeds so differences are attributable to
the single change. Ablations covered:
  full                    -- everything on
  no_world_model          -- skip Stage-C imagined search (use base grounded plan)
  every_step_planning     -- replan every control step instead of on events
  no_adaptation           -- freeze DynamicsContext (no online estimation)
  no_recovery             -- never replan on failure
  absolute_transforms     -- compile with absolute (world) place targets, not
                             relative-to-reference (breaks one-shot transfer)
  privileged_state        -- oracle estimator (perfect state) instead of belief
"""
from __future__ import annotations

from dataclasses import replace

from ..execution.loop import ExecConfig
from ..metrics.metrics import aggregate
from .runner import run_benchmark


def _absolute_transform_graphs():
    """Rewrite each place transition to target the demo's ABSOLUTE world pose
    (reference=world), discarding the object-relative frame. At eval the layout is
    randomized, so an absolute target lands in the wrong place -- this is the
    relative-vs-absolute ablation and should show a large success drop."""
    from ..tasks import DEFAULT_TASKS, record_demo
    graphs = {}
    for t in DEFAULT_TASKS:
        g = record_demo(t)
        for tr in g.transitions:
            if tr.reason == "place" and tr.abs_target is not None:
                tr.reference = "world"
                tr.rel_transform = tr.abs_target.copy()
        # goal still references roles; keep on_top/at_rel checks relative so the
        # ablation isolates the PLAN target frame, not the success criterion.
        graphs[t.name] = g
    return graphs


def run_ablations(seeds=range(20)):
    rows = {}
    base = ExecConfig()
    configs = {
        "full": base,
        "no_world_model": replace(base, use_world_model=False),
        "every_step_planning": replace(base, event_driven=False),
        "no_adaptation": replace(base, adapt=False),
        "no_recovery": replace(base, allow_recovery=False),
    }
    for name, cfg in configs.items():
        recs = run_benchmark(seeds=seeds, cfg=cfg)
        rows[name] = aggregate(recs)

    # retargeting: absolute vs raw-relative vs semantic (the central-thesis ablation)
    for mode in ("absolute", "relative", "semantic"):
        rows[f"retarget_{mode}"] = aggregate(
            run_benchmark(seeds=seeds, cfg=replace(base, retarget_mode=mode)))

    # privileged-state ablation (oracle estimator)
    rows["privileged_state"] = aggregate(
        run_benchmark(seeds=seeds, cfg=base, privileged=True))

    return rows


def format_table(rows: dict) -> str:
    hdr = f"{'ablation':22s} {'success':>8s} {'first':>7s} {'wrong_belief':>13s} {'plan_p95_ms':>12s}"
    lines = [hdr, "-" * len(hdr)]
    for name, r in rows.items():
        lines.append(f"{name:22s} {r.success_rate:7.1%} {r.first_attempt_rate:6.1%} "
                     f"{r.wrong_belief_rate:12.1%} {r.plan_latency_ms['p95']:11.2f}")
    return "\n".join(lines)
