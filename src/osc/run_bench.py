"""Full benchmark: splits x tasks x seed groups, with JSON/MD/failure outputs.

    python -m osc.run_bench --seeds 20 --out reports/v0_2

Multiple independent seed groups + bootstrap CIs; no claim rests on one 30-episode
seed. Ground truth is read only by the Scorer.
"""
from __future__ import annotations

import argparse
import os

from .benchmark.runner import run_benchmark, write_reports
from .execution.loop import ExecConfig
from .metrics.metrics import aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-groups", type=int, default=1,
                    help="run this many disjoint seed blocks for CI stability")
    ap.add_argument("--out", default="reports/v0_2")
    args = ap.parse_args()

    all_records = []
    for g in range(args.seed_groups):
        seeds = range(g * args.seeds, (g + 1) * args.seeds)
        all_records += run_benchmark(seeds=seeds, cfg=ExecConfig())

    report = aggregate(all_records)
    print(report.pretty())
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prefix = write_reports(all_records, report, args.out)
    print(f"\nwrote {prefix}.json / .md / .failures.jsonl")


if __name__ == "__main__":
    main()
