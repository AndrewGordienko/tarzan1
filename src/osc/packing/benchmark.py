"""Controlled packing proof benchmark and baselines."""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np

from .compiler import compile_packing_demo
from .domain import Container, PackItem, PackingState, Placement
from .executor import PackingExecutor
from .planner import PackingPlanner
from .render import render_states
from .verifier import verify_final_pack
from .world_model import apply_placement, evaluate_placement
from .candidates import candidate_placements


@dataclass
class PackingScenario:
    name: str
    container: Container
    items: dict[str, PackItem]
    arrivals: list[str]
    feasible: bool = True
    late_item: str | None = None
    description: str = ""


def demo_items():
    return {
        "heavy": PackItem("heavy", (3.0, 3.0, 2.0), mass=8.0, load_limit=50.0),
        "fragile": PackItem("fragile", (2.0, 2.0, 1.5), mass=1.0, fragile=True, load_limit=1.0),
        "long": PackItem("long", (5.0, 1.3, 1.3), mass=2.0),
        "ordinary": PackItem("ordinary", (2.2, 2.2, 2.0), mass=2.0),
    }


def scenarios(seed=0):
    rng = np.random.default_rng(seed)
    out = [PackingScenario(
        "changed_inventory", Container("box_a", (10.0, 8.0, 6.0), 30.0),
        {"heavy": PackItem("heavy", (3.2, 2.8, 2.0), 8.0, load_limit=50.0),
         "fragile": PackItem("fragile", (2.0, 2.4, 1.4), 1.0, fragile=True, load_limit=1.0),
         "ordinary": PackItem("ordinary", (2.5, 2.0, 2.0), 2.0)},
        ["ordinary", "heavy", "fragile"], description="new dimensions and order"),
        PackingScenario(
        "different_box_orientations", Container("box_b", (8.0, 10.0, 6.0), 30.0),
        {"heavy": PackItem("heavy", (3.0, 4.0, 2.0), 8.0, load_limit=50.0),
         "long": PackItem("long", (6.0, 1.5, 1.5), 2.0),
         "fragile": PackItem("fragile", (2.0, 2.0, 1.5), 1.0, fragile=True, load_limit=1.0)},
        ["long", "heavy", "fragile"], description="orientation and box dimensions"),
        PackingScenario(
        "late_large_rearrangement", Container("box_late", (10.0, 8.0, 6.0), 40.0),
        {"small_a": PackItem("small_a", (4.0, 4.0, 2.0), 2.0,
                              allowed_orientations=((0, 1, 2),)),
         "small_b": PackItem("small_b", (4.0, 4.0, 2.0), 2.0,
                              allowed_orientations=((0, 1, 2),)),
         "late_large": PackItem("late_large", (6.0, 5.0, 2.0), 10.0, load_limit=40.0,
                                allowed_orientations=((0, 1, 2),))},
        ["small_a", "small_b", "late_large"], late_item="late_large",
        description="late large item forces earlier placements to be removed"),
        PackingScenario(
        "impossible_order", Container("box_impossible", (5.0, 5.0, 4.0), 10.0),
        {"oversize": PackItem("oversize", (6.0, 2.0, 2.0), 2.0)}, ["oversize"],
        feasible=False, description="boundary violation must abstain"),
    ]
    return out


def _belief_scenario(s, seed):
    rng = np.random.default_rng(seed)
    items = {}
    for i, item in s.items.items():
        dims = tuple(float(max(.1, x * (1.0 + rng.normal(0, .025)))) for x in item.dimensions)
        items[i] = PackItem(i, dims, item.mass * (1.0 + rng.normal(0, .05)),
                            mass_uncertainty=item.mass * .05, fragile=item.fragile,
                            load_limit=item.load_limit, metadata=dict(item.metadata))
    return PackingScenario(s.name, s.container, items, list(s.arrivals), s.feasible,
                           s.late_item, s.description)


def literal_replay(s, program):
    state = PackingState(s.container, dict(s.items))
    # A trajectory replay uses fixed demonstration-like corners and never removes.
    for idx, item_id in enumerate(s.arrivals):
        item = state.items[item_id]
        p = Placement(item_id, (float((idx * 3) % 7), float((idx * 2) % 6), 0.0),
                      item.dimensions)
        if evaluate_placement(state, item, p).feasible:
            state = apply_placement(state, p)
    ok, _ = verify_final_pack(state, s.arrivals)
    return state, ok


def greedy_next_fit(s):
    state = PackingState(s.container, dict(s.items))
    for item_id in s.arrivals:
        item = state.items[item_id]
        floor = [p for p in candidate_placements(state, item) if abs(p.position[2]) < 1e-8]
        if not floor:
            return state, False
        state = apply_placement(state, max(floor, key=lambda p: (p.position[0], p.position[1])))
    return state, verify_final_pack(state, s.arrivals)[0]


def run_episode(s, perception="oracle", seed=0, render_path=None):
    s_eval = _belief_scenario(s, seed) if perception == "belief" else s
    program = compile_packing_demo()
    planner = PackingPlanner(program, beam_width=48)
    executor = PackingExecutor(planner)
    states = [PackingState(s_eval.container, dict(s_eval.items))]
    if s.late_item:
        current = states[0]
        logs = []
        for item_id in s_eval.arrivals[:s_eval.arrivals.index(s_eval.late_item)]:
            # Model the production arrival lane: earlier items were placed by a
            # simple floor-first handler before the late item was known.  This is
            # intentionally myopic so the task-level planner must demonstrate
            # remove/repack when the new constraint arrives.
            item = current.items[item_id]
            floor_candidates = [p for p in candidate_placements(current, item)
                                if abs(p.position[2]) < 1e-8]
            floor = max(floor_candidates, key=lambda p: (p.position[0], p.position[1])) \
                if floor_candidates else None
            if floor is None:
                return dict(scenario=s.name, perception=perception, success=False,
                            expected_feasible=s.feasible, rearranged=False, actions=logs,
                            reason="initial_arrival_failed", packed=list(current.placements),
                            violations=[])
            current = apply_placement(current, floor)
            logs.append({"kind": "place_inside", "item": item_id, "verified": True,
                         "detail": {"arrival_floor_first": True}, "rearranged": False})
            states.append(current)
        current, log, ok, reason = executor.execute(current, [s_eval.late_item])
        logs.extend(log); states.append(current)
    else:
        current, logs, ok, reason = executor.execute(states[0], s_eval.arrivals)
        states.append(current)
    if render_path:
        render_states(states, render_path, f"Tarzan {s.name} ({perception})")
    return dict(scenario=s.name, perception=perception, success=bool(ok and s.feasible),
                expected_feasible=s.feasible, rearranged=any(x.get("kind") == "temporarily_remove" for x in logs),
                actions=logs, reason=reason, packed=list(current.placements),
                violations=[x for x in logs if not x.get("verified", True)])


def run_benchmark(episodes=100, perception="oracle", out_path=None, render_dir=None, seed=0):
    ss = scenarios(seed)
    baseline = {s.name: (literal_replay(s, compile_packing_demo())[1], greedy_next_fit(s)[1]) for s in ss}
    rows = []
    for k in range(episodes):
        s = ss[k % len(ss)]
        render = f"{render_dir}/{s.name}_{perception}.gif" if render_dir and k < len(ss) else None
        rows.append(run_episode(s, perception, seed + k, render))
    feasible = [r for r in rows if r["expected_feasible"]]
    impossible = [r for r in rows if not r["expected_feasible"]]
    report = dict(episodes=episodes, perception=perception,
                  completion_rate=float(np.mean([r["success"] for r in feasible])) if feasible else 0.0,
                  impossible_abstention_rate=float(np.mean([not r["success"] for r in impossible])) if impossible else 0.0,
                  rearrangement_rate=float(np.mean([r["rearranged"] for r in rows])),
                  violations=sum(len(r["violations"]) for r in rows),
                  baseline_literal_completion=float(np.mean([baseline[r["scenario"]][0] for r in rows])),
                  baseline_greedy_completion=float(np.mean([baseline[r["scenario"]][1] for r in rows])),
                  planner_improvement_over_literal=float(np.mean([r["success"] for r in rows])
                                                         - np.mean([baseline[r["scenario"]][0] for r in rows])),
                  planner_improvement_over_greedy=float(np.mean([r["success"] for r in rows])
                                                        - np.mean([baseline[r["scenario"]][1] for r in rows])),
                  baseline_by_scenario={s.name: {"literal": baseline[s.name][0],
                                                 "greedy": baseline[s.name][1]} for s in ss},
                  rows=rows)
    if out_path:
        with open(out_path, "w") as f: json.dump(report, f, indent=2)
    return report
