"""Paired resolution-scenario ablation: does the policy pick the action that
actually carries information, and refuse to fake genuine ambiguity?

For each scenario, the EXPECTED resolving capability is:
  noisy_identifiable -> inspection      occluded    -> viewpoint
  interaction        -> probe           fundamental -> clarification/metadata
"""
from __future__ import annotations

import argparse

from .benchmark.resolution_scenarios import CONFIGS, SCENARIOS, run_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    args = ap.parse_args()
    res = run_scenarios(seeds=range(args.seeds))
    cnames = list(CONFIGS)
    w = 15
    print("=" * (18 + w * len(cnames)))
    print("  RESOLUTION SCENARIOS  -- 'autonomous-correct% / human%' per capability")
    print("=" * (18 + w * len(cnames)))
    print(f"{'scenario':18s}" + "".join(f"{c:>{w}}" for c in cnames))
    print("-" * (18 + w * len(cnames)))
    for kind in SCENARIOS:
        row = f"{kind:18s}"
        for c in cnames:
            d = res[kind][c]
            row += f"{100*d['autonomous_correct']:5.0f}/{100*d['human']:<3.0f}h   "
        print(row)
    print("-" * (18 + w * len(cnames)))
    print("  read: left = % resolved autonomously & correctly; right = % needed a human.")
    print("  gate: inspection resolves noisy; viewpoint -> occluded; probe -> interaction;")
    print("        fundamental needs a human under every physical capability.")


if __name__ == "__main__":
    main()
