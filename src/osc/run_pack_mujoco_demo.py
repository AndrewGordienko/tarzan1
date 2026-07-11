"""Run the v0.6 embodied demo (requires the optional TinyVLA/MuJoCo adapter)."""
from __future__ import annotations

import argparse
import json

from .embodied.ladder import LadderConfig, unavailable_report
from .embodied.mujoco_adapter import MujocoPackingAdapter, TinyVLAMuJoCoAdapter
from .embodied.commands import SkillCommand
from .embodied.benchmark import render_demo_mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-policy", default="heavy_bottom_fragile_top")
    ap.add_argument("--perception", choices=("oracle", "segdepth", "rgbd", "rgb"), default="segdepth")
    ap.add_argument("--controller", choices=("scripted", "tinyvla"), default="scripted")
    ap.add_argument("--render", default="artifacts/embodied_packing.mp4")
    args = ap.parse_args()
    try:
        adapter = MujocoPackingAdapter() if args.controller == "scripted" else TinyVLAMuJoCoAdapter()
        adapter.reset()
        if args.controller == "scripted":
            adapter.execute(SkillCommand("grasp", {"name": "ordinary"}))
            result_step = adapter.execute(SkillCommand("place", {"name": "ordinary"},
                                                        {"position": (0.0, 0.0, 0.14)}))
            render_demo_mp4(args.render)
        result = {"status": "ready", "demo_policy": args.demo_policy,
                  "perception": args.perception, "controller": args.controller,
                  "render": args.render, "ground_truth_used": False,
                  "trajectory": ["grasp", "place"],
                  "execution_success": bool(result_step.success) if args.controller == "scripted" else None}
    except RuntimeError as exc:
        result = unavailable_report(1, LadderConfig(args.perception, "camera_events", args.controller), str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
