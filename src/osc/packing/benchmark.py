"""Controlled packing proof benchmark and baselines."""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np

from .compiler import (compile_packing_demo, compile_with_inferred_posterior,
                       compile_composed_program, infer_program_posterior)
from .demonstration import PackingEvent
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


def policy_sensitive_scenario():
    """One fixed inventory used to visualize different customer policies."""
    return PackingScenario(
        "policy_sensitive_same_order", Container("box_policy", (8.0, 8.0, 6.0), 30.0),
        {"heavy": PackItem("heavy", (3.0, 3.0, 2.0), 8.0, load_limit=50.0),
         "fragile": PackItem("fragile", (3.0, 3.0, 1.5), 1.0, fragile=True, load_limit=1.0),
         "ordinary": PackItem("ordinary", (2.0, 2.0, 2.0), 2.0),
         "long": PackItem("long", (5.0, 1.5, 1.5), 2.0)},
        ["ordinary", "long", "heavy", "fragile"], description="same inventory, policy changes")


def policy_demos():
    return {
        "correct_heavy_bottom": [PackingEvent("place_inside", "heavy", properties={"policy": "heavy_bottom_fragile_top"}),
                                 PackingEvent("place_inside", "fragile"),
                                 PackingEvent("place_inside", "ordinary")],
        "different_max_volume": [PackingEvent("place_inside", "long", properties={"fill": "high"}),
                                  PackingEvent("place_inside", "ordinary"),
                                  PackingEvent("place_inside", "heavy"),
                                  PackingEvent("place_inside", "fragile")],
        "minimize_rehandling": [PackingEvent("place_inside", "ordinary", properties={"rehandling": "low"}),
                                PackingEvent("place_inside", "long"),
                                PackingEvent("place_inside", "heavy"),
                                PackingEvent("place_inside", "fragile")],
        "conflicting": [PackingEvent("place_inside", "fragile"),
                         PackingEvent("place_inside", "heavy"),
                         PackingEvent("rearrange", "heavy"),
                         PackingEvent("place_inside", "ordinary")],
    }


def _belief_scenario(s, seed):
    rng = np.random.default_rng(seed)
    items = {}
    for i, item in s.items.items():
        dims = tuple(float(max(.1, x * (1.0 + rng.normal(0, .025)))) for x in item.dimensions)
        items[i] = PackItem(i, dims, item.mass * (1.0 + rng.normal(0, .05)),
                            mass_uncertainty=item.mass * .05, fragile=item.fragile,
                            load_limit=item.load_limit, allowed_orientations=item.allowed_orientations,
                            metadata=dict(item.metadata))
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


def _violation_counts(state, s):
    ok, rows = verify_final_pack(state, s.arrivals)
    counts = {"geometric": 0, "load": 0, "fragility": 0,
              "false_completion": int(ok), "correct_abstention": int(not ok and not s.feasible)}
    for detail_ok, detail in rows.values():
        if not detail_ok:
            if detail.get("reason") == "missing" or detail.get("collision") or detail.get("boundary") or detail.get("support", 1.0) < .85:
                counts["geometric"] += 1
            if detail.get("load_violation"):
                counts["load"] += 1
            if detail.get("fragility_violation"):
                counts["fragility"] += 1
    return counts


def run_episode(s, perception="oracle", seed=0, render_path=None, program=None, capture_states=False):
    s_eval = _belief_scenario(s, seed) if perception == "belief" else s
    program = program or compile_packing_demo()
    if program.policy_name == "unknown_or_unexplained":
        return dict(scenario=s.name, perception=perception, success=False,
                    expected_feasible=s.feasible, rearranged=False, actions=[],
                    reason="unknown_program_abstain", packed=[], placement_layout={},
                    violations=[])
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
    result = dict(scenario=s.name, perception=perception, success=bool(ok and s.feasible),
                expected_feasible=s.feasible, rearranged=any(x.get("kind") == "temporarily_remove" for x in logs),
                actions=logs, reason=reason, packed=list(current.placements),
                placement_layout={i: {"position": p.position, "size": p.size}
                                  for i, p in current.placements.items()},
                violations=[x for x in logs if not x.get("verified", True)])
    if capture_states:
        result["_states"] = states
    return result


def run_demo_dependence(perception="oracle", seed=0):
    s = policy_sensitive_scenario()
    demos = policy_demos()
    rows = {}
    for name, events in demos.items():
        program = compile_with_inferred_posterior(events)
        result = run_episode(s, perception, seed, program=program, capture_states=True)
        result["program_policy"] = program.policy_name
        result["program_posterior"] = program.posterior
        result["constraint_posterior"] = program.constraint_posterior
        result["policy_behavior_match"] = _policy_behavior_match(program.policy_name, result)
        result["arrangement"] = result.get("placement_layout", result["packed"])
        rows[name] = result
    # Controls: no demo, temporal shuffle, and oracle task program.
    no_demo = compile_packing_demo([], "neutral_prior")
    rows["no_demo"] = run_episode(s, perception, seed, program=no_demo)
    rows["no_demo"]["arrangement"] = rows["no_demo"].get("placement_layout", rows["no_demo"]["packed"])
    rows["no_demo"]["program_policy"] = no_demo.policy_name
    rows["no_demo"]["policy_behavior_match"] = _policy_behavior_match(no_demo.policy_name, rows["no_demo"])
    shuffled = [PackingEvent(e.kind, e.item_id) for e in reversed(demos["correct_heavy_bottom"])]
    shuffled_program = compile_with_inferred_posterior(shuffled)
    rows["shuffled_demo"] = run_episode(s, perception, seed, program=shuffled_program)
    rows["shuffled_demo"]["arrangement"] = rows["shuffled_demo"].get("placement_layout", rows["shuffled_demo"]["packed"])
    rows["shuffled_demo"]["program_policy"] = shuffled_program.policy_name
    rows["shuffled_demo"]["program_posterior"] = shuffled_program.posterior
    rows["shuffled_demo"]["policy_behavior_match"] = _policy_behavior_match(shuffled_program.policy_name, rows["shuffled_demo"])
    oracle = compile_packing_demo([], "heavy_bottom_fragile_top")
    rows["oracle_task_program"] = run_episode(s, perception, seed, program=oracle)
    rows["oracle_task_program"]["arrangement"] = rows["oracle_task_program"].get("placement_layout", rows["oracle_task_program"]["packed"])
    rows["oracle_task_program"]["program_policy"] = oracle.policy_name
    rows["oracle_task_program"]["policy_behavior_match"] = _policy_behavior_match(oracle.policy_name, rows["oracle_task_program"])
    return rows


def run_forced_rearrangement_intervention(perception="oracle", seed=0):
    """Matched intervention: both lanes receive the identical packed prefix, then
    the late large item is revealed.  Completion requires remove->replan->repack."""
    scenario = scenarios(seed)[2]
    estimated = _belief_scenario(scenario, seed) if perception == "belief" else scenario
    current = PackingState(scenario.container, dict(estimated.items))
    # Prefix is fixed from ground-truth geometry, so perception cannot choose a
    # more convenient initial arrangement in one lane.
    for item_id, pos in (("small_a", (0.0, 0.0, 0.0)), ("small_b", (4.0, 0.0, 0.0))):
        item = current.items[item_id]
        current = apply_placement(current, Placement(item_id, pos, item.dimensions,
                                                     (0, 1, 2)))
    before = dict(current.placements)
    planner = PackingPlanner(compile_packing_demo(), beam_width=48)
    current, log, ok, reason = PackingExecutor(planner).execute(current, ["late_large"])
    removed = [x for x in log if x.get("kind") == "temporarily_remove"]
    return dict(perception=perception, success=bool(ok), required_rearrangement=True,
                rearrangement_occurred=bool(removed), remove_count=len(removed),
                action_kinds=[x.get("kind") for x in log], reason=reason,
                initial_prefix={i: p.position for i, p in before.items()},
                final_layout={i: p.position for i, p in current.placements.items()})


def run_heldout_composition(perception="oracle", seed=0):
    s = policy_sensitive_scenario()
    program = compile_composed_program(("heavy_below_fragile", "minimize_rehandling"))
    result = run_episode(s, perception, seed, program=program)
    result["heldout_components"] = ["heavy_below_fragile", "minimize_rehandling"]
    result["program_policy"] = program.policy_name
    return result


def _policy_behavior_match(policy_name, result):
    layout = result.get("arrangement", result.get("placement_layout", {}))
    if not layout:
        return False
    fragile_z = layout.get("fragile", {}).get("position", (0, 0, 0))[2]
    if policy_name == "heavy_bottom_fragile_top":
        return fragile_z <= 1e-8
    if policy_name == "maximize_volume":
        return fragile_z > 1e-8
    if policy_name == "minimize_rehandling":
        return not result.get("rearranged", False)
    return False


def run_benchmark(episodes=100, perception="oracle", out_path=None, render_dir=None, seed=0):
    ss = scenarios(seed)
    baseline = {}
    for s in ss:
        ls, lok = literal_replay(s, compile_packing_demo())
        gs, gok = greedy_next_fit(s)
        baseline[s.name] = dict(literal=lok, greedy=gok,
                                literal_violations=_violation_counts(ls, s),
                                greedy_violations=_violation_counts(gs, s))
    rows = []
    for k in range(episodes):
        s = ss[k % len(ss)]
        render = f"{render_dir}/{s.name}_{perception}.gif" if render_dir and k < len(ss) else None
        rows.append(run_episode(s, perception, seed + k, render))
    feasible = [r for r in rows if r["expected_feasible"]]
    impossible = [r for r in rows if not r["expected_feasible"]]
    dep = run_demo_dependence(perception, seed)
    report = dict(episodes=episodes, perception=perception,
                  completion_rate=float(np.mean([r["success"] for r in feasible])) if feasible else 0.0,
                  impossible_abstention_rate=float(np.mean([not r["success"] for r in impossible])) if impossible else 0.0,
                  overall_correct_decision=float(np.mean([r["success"] if r["expected_feasible"]
                                                         else not r["success"] for r in rows])),
                  rearrangement_rate=float(np.mean([r["rearranged"] for r in rows])),
                  violations=sum(len(r["violations"]) for r in rows),
                  baseline_literal_completion=float(np.mean([baseline[r["scenario"]]["literal"] for r in rows])),
                  baseline_greedy_completion=float(np.mean([baseline[r["scenario"]]["greedy"] for r in rows])),
                  baseline_overall_correct_decision={
                      "literal": float(np.mean([baseline[r["scenario"]]["literal"] if r["expected_feasible"]
                                                 else not baseline[r["scenario"]]["literal"] for r in rows])),
                      "greedy": float(np.mean([baseline[r["scenario"]]["greedy"] if r["expected_feasible"]
                                                else not baseline[r["scenario"]]["greedy"] for r in rows]))},
                  planner_improvement_over_literal=float(np.mean([r["success"] for r in rows])
                                                         - np.mean([baseline[r["scenario"]]["literal"] for r in rows])),
                  planner_improvement_over_greedy=float(np.mean([r["success"] for r in rows])
                                                        - np.mean([baseline[r["scenario"]]["greedy"] for r in rows])),
                  baseline_by_scenario={s.name: {"literal": baseline[s.name]["literal"],
                                                 "greedy": baseline[s.name]["greedy"],
                                                 "literal_violations": baseline[s.name]["literal_violations"],
                                                 "greedy_violations": baseline[s.name]["greedy_violations"]} for s in ss},
                  demo_dependence={k: {x: v[x] for x in ("success", "program_policy", "program_posterior",
                                                         "constraint_posterior", "policy_behavior_match", "arrangement")
                                      if x in v} for k, v in dep.items()},
                  forced_rearrangement={p: run_forced_rearrangement_intervention(p, seed)
                                        for p in ("oracle", "belief")},
                  heldout_composition={p: run_heldout_composition(p, seed)
                                       for p in ("oracle", "belief")},
                  rows=rows)
    if out_path:
        with open(out_path, "w") as f: json.dump(report, f, indent=2)
    return report
