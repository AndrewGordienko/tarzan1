"""Run the exact-assignment correspondence isolation ladder."""
from __future__ import annotations

import argparse
import json

from .benchmark.correspondence_audit import run_correspondence_isolation
from .benchmark.runner import _json_default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default="correspondence_isolation.json")
    args = ap.parse_args()
    report = run_correspondence_isolation(seeds=range(args.seeds))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    for name, row in report["summary"].items():
        print(f"{name:48s} top={row['top_assignment_correct_rate']:.3f} "
              f"present={row['gt_assignment_present_rate']:.3f} "
              f"rank={row['gt_rank_first_rate']:.3f} "
              f"silent={row['silent_committed_binding_error']:.3f} "
              f"null={row['null_mass_mean']:.3f} outsideK={row['outside_top_k_mass_mean']:.3f}")


if __name__ == "__main__":
    main()
