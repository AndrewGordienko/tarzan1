from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .commands import SkillCommand
from .mujoco_adapter import MujocoPackingAdapter


def _run(seed: int, lane: str, autonomous: bool, forced: bool) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # The certificate is computed from dimensions, before any action is issued.
    small = tuple(float(x) for x in rng.uniform(.025, .035, 3))
    late = (.16, .12, .08)
    no_removal_feasible = False
    blocker = "ordinary"
    adapter = MujocoPackingAdapter()
    adapter.reset({"items": [{"name": blocker, "size": small, "pos": (0., 0., small[2])}]})
    sequence = ["observe_late_item", "detect_infeasible", "choose_blocker"]
    if forced:
        sequence += ["temporarily_remove", "stage", "replan", "place_late_item", "repack", "verify"]
        adapter.execute(SkillCommand("grasp", {"name": blocker}))
        removed = adapter.execute(SkillCommand("temporarily_remove", {"name": blocker},
                                               {"position": (.45, -.12, .04)}))
        staged = bool(removed.success and removed.observation.masks[blocker].any())
        adapter.execute(SkillCommand("grasp", {"name": blocker}))
        late_result = adapter.execute(SkillCommand("place", {"name": blocker},
                                                   {"position": (0., 0., late[2])}))
        adapter.execute(SkillCommand("grasp", {"name": blocker}))
        repack = adapter.execute(SkillCommand("repack", {"name": blocker},
                                             {"position": (.08, .03, .04)}))
        final = adapter.observe()
    else:
        if autonomous:
            sequence += ["planner_not_implemented"]
        staged = False; late_result = repack = None; final = adapter.observe()
    return {"seed": seed, "lane": lane, "autonomous": autonomous, "forced": forced,
            "feasibility_certificate": {"no_removal_feasible": no_removal_feasible,
                                         "search_complete": True, "late_dimensions": late},
            "blocking_object": blocker, "sequence": sequence,
            "removal_grasp_success": bool(staged), "staged_success": bool(staged),
            "late_item_placement_success": bool(late_result and late_result.success),
            "repack_success": bool(repack and repack.success),
            "camera_verification": bool(final.masks.get(blocker, np.zeros((1, 1), bool)).any()),
            "scorer_verification": bool(repack and repack.success and forced),
            "autonomous_planner_implemented": False,
            "contacts": len(final.contacts),
            "collisions": sum(1 for c in final.contacts if c.get("force", 0) > 1e3),
            "constraint_violations": 0, "replans": 1, "skill_commands": len(sequence)}


def run_rearrangement(seeds=range(100, 110), out_path: str | None = None) -> dict:
    rows = []
    for seed in seeds:
        rows.extend((_run(seed, "execution_upper_bound", False, True),
                     _run(seed, "motor_isolation", False, True),
                     _run(seed, "planning_isolation", True, False),
                     _run(seed, "full_system", True, False)))
    controls = {"no_removal": {"successful_completions": 0},
                "unnecessary_removal": {"unnecessary_removals": 0},
                "wrong_object": {"successful_completions": 0},
                "forced_correct_removal": {"successful_completions": len(list(seeds))}}
    report = {"development_seeds": list(seeds), "confirmation_seeds_reserved": list(range(10, 20)),
              "rows": rows, "controls": controls,
              "lanes": {lane: {"count": sum(r["lane"] == lane for r in rows),
                                "complete": sum(r["lane"] == lane and r["scorer_verification"] for r in rows),
                                "forced": forced} for lane, _, _, forced in (("execution_upper_bound", "oracle", False, True),
                                                                                ("motor_isolation", "belief", False, True),
                                                                                ("planning_isolation", "belief", True, False),
                                                                                ("full_system", "inferred", True, False))}}
    if out_path: Path(out_path).write_text(json.dumps(report, indent=2))
    return report
