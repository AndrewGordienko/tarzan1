"""Run the ablation suite and print the component-attribution table.

    python -m osc.run_ablations --seeds 20
"""
from __future__ import annotations

import argparse

from .benchmark.ablations import format_table, run_ablations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    rows = run_ablations(seeds=range(args.seeds))
    print("\nABLATIONS (same tasks/splits/seeds; one change each)\n")
    print(format_table(rows))


if __name__ == "__main__":
    main()
