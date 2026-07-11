"""Compile and render the canonical packing demonstration."""
from __future__ import annotations

import argparse
from .packing.compiler import compile_with_inferred_posterior
from .packing.benchmark import scenarios, run_episode, run_demo_dependence
from .packing.render import render_policy_comparison


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="artifacts/packing_demo.gif")
    ap.add_argument("--program", default="artifacts/packing_program.json")
    ap.add_argument("--comparison", default="artifacts/packing_policy_comparison.gif")
    args = ap.parse_args()
    program = compile_with_inferred_posterior(); program.save(args.program)
    run_episode(scenarios()[0], perception="oracle", render_path=args.render)
    rows = run_demo_dependence("oracle")
    late = run_episode(scenarios()[2], "oracle", capture_states=True)
    render_policy_comparison(rows, late, args.comparison)
    print(args.render)
    print(args.program)
    print(args.comparison)


if __name__ == "__main__": main()
