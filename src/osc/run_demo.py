"""End-to-end vertical slice: one demonstration -> compiled program -> execution
across randomized environments with an injected disturbance, no fine-tuning.

    python -m osc.run_demo --episodes 30 --seed 0

Stages exercised:
  A  compile_demo         : demo tracks -> object-centric task graph
  B  ground_plan          : task graph -> reusable skill experts (router)
  C  ImaginedSearch       : score candidate plans in the world model
  D  ClosedLoopExecutor   : execute, adapt online, replan only on events
"""
from __future__ import annotations

import argparse

import numpy as np

from .compiler.stage_a import compile_demo
from .execution.loop import ClosedLoopExecutor
from .metrics.metrics import aggregate
from .sim.disturbance import sample_disturbance
from .sim.randomize import RandomizationSpec, randomize
from .skills.grounding import ground_plan
from .tasks import STACK_SCENE, record_demo
from .worldmodel.model import WorldModel
from .worldmodel.search import ImaginedSearch


def build_program(verbose: bool = True):
    """Stage A (+ show grounded Stage B plan). Runs exactly once, no weights."""
    trace = record_demo(STACK_SCENE)
    graph = compile_demo(trace, STACK_SCENE["roles"])
    if verbose:
        print("\n--- Stage A: compiled task graph (from ONE demonstration) ---")
        print(graph.pretty())
        print("\n--- Stage B: grounded skill sequence (the router's picks) ---")
        for si in ground_plan(graph):
            print(f"    {si.label}")
    return graph


def run(episodes: int = 30, seed: int = 0, disturb: bool = True, verbose: bool = True):
    graph = build_program(verbose=verbose)
    spec = RandomizationSpec()
    results = []
    for i in range(episodes):
        ep_seed = seed * 10_000 + i
        state, backend = randomize(STACK_SCENE, spec, seed=ep_seed)
        if disturb:
            # horizon ~ typical episode length so the disturbance lands mid-run.
            dist = sample_disturbance(list(state.objects.keys()),
                                      horizon=16, seed=ep_seed)
            backend._pre_step_hook = dist
        backend.reset(state)

        wm = WorldModel(ensemble_size=5, seed=ep_seed)
        search = ImaginedSearch(wm)
        executor = ClosedLoopExecutor(backend, graph, search)
        res = executor.run()
        results.append(res)
        if verbose:
            tag = "OK " if res.success else "FAIL"
            print(f"  ep{i:03d} seed={ep_seed:<7} {tag} "
                  f"replans={res.replans} steps={res.steps} "
                  f"plan={res.mean_plan_latency_ms:5.1f}ms "
                  f"safety={res.safety_violations}")

    report = aggregate(results, demos_required=1)
    print("\n" + report.pretty())
    return report, results


def main():
    ap = argparse.ArgumentParser(description="One-Shot Task Compiler -- vertical slice")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-disturb", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    np.seterr(all="ignore")
    run(episodes=args.episodes, seed=args.seed,
        disturb=not args.no_disturb, verbose=not args.quiet)


if __name__ == "__main__":
    main()
