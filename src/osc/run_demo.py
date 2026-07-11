"""Single-task demo of the v0.2 belief-state pipeline (quick sanity view).

    python -m osc.run_demo --task stack --episodes 12

Prints the compiled task graph (Stage A, role-based), then runs a few belief-
state episodes under the disturbance-recovery split and shows per-episode outcome
plus a small aggregate. For the full split x task x seed benchmark use
`python -m osc.run_bench`.
"""
from __future__ import annotations

import argparse

from .benchmark.runner import Split, default_splits, run_episode
from .execution.loop import ExecConfig
from .metrics.metrics import aggregate
from .tasks import TASKS, record_demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="stack", choices=list(TASKS))
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--split", default="disturbance_recovery")
    args = ap.parse_args()

    task = TASKS[args.task]
    graph = record_demo(task)
    print("\n--- Stage A: role-based task graph (from ONE demonstration) ---")
    print(graph.pretty())

    split = next(s for s in default_splits() if s.name == args.split)
    records = [run_episode(task, graph, split, seed, ExecConfig())
               for seed in range(args.episodes)]
    for r in records:
        tag = "OK " if r.success else f"FAIL[{r.failure_category}]"
        print(f"  seed={r.seed:<3} {tag:14s} replans={r.autonomous_replans} "
              f"steps={r.steps} wrong_belief={int(r.wrong_belief)} "
              f"plan_p95={_p95(r.plan_latencies_ms):.1f}ms")
    print("\n" + aggregate(records).pretty())


def _p95(xs):
    import numpy as np
    return float(np.percentile(xs, 95)) if xs else 0.0


if __name__ == "__main__":
    main()
