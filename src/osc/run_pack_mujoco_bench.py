"""Run the v0.6 embodied benchmark; dependency failures are explicit."""
from __future__ import annotations

import argparse
import json

from .embodied.ladder import LadderConfig, unavailable_report
from .embodied.mujoco_adapter import MujocoPackingAdapter, TinyVLAMuJoCoAdapter
from .embodied.benchmark import run_attribution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--perception", choices=("oracle", "segdepth", "rgbd", "rgb"), default="segdepth")
    ap.add_argument("--controller", choices=("scripted", "tinyvla"), default="scripted")
    args = ap.parse_args()
    try:
        (MujocoPackingAdapter() if args.controller == "scripted" else TinyVLAMuJoCoAdapter()).reset()
        result = run_attribution(min(args.episodes, 10))
        result.update({"status": "ready", "episodes": args.episodes, "perception": args.perception,
                       "controller": args.controller, "ground_truth_used": False})
    except RuntimeError as exc:
        result = unavailable_report(args.episodes, LadderConfig(args.perception, "camera_events", args.controller), str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
