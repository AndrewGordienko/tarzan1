"""Run the paired oracle attribution ladder and print the error budget.

    python -m osc.run_attribution --seeds 20

This is the v0.3 headline: it turns the circular 'perception' label into an
evidence-based percentage-point budget by swapping in a perfect version of one
component at a time (same task/split/seed).
"""
from __future__ import annotations

import argparse

from .benchmark.attribution import error_budget, run_ladder
from .benchmark.seeds import dev_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--heldout", action="store_true", help="use held-out seed block")
    args = ap.parse_args()
    seeds = dev_seeds(args.seeds, heldout=args.heldout)
    print(error_budget(run_ladder(seeds=seeds)))


if __name__ == "__main__":
    main()
