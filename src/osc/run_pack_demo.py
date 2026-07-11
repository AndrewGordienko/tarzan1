"""Compile and render the canonical packing demonstration."""
from __future__ import annotations

import argparse
from .packing.compiler import compile_with_inferred_posterior
from .packing.benchmark import scenarios, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", default="artifacts/packing_demo.gif")
    ap.add_argument("--program", default="artifacts/packing_program.json")
    args = ap.parse_args()
    program = compile_with_inferred_posterior(); program.save(args.program)
    run_episode(scenarios()[0], perception="oracle", render_path=args.render)
    print(args.render)
    print(args.program)


if __name__ == "__main__": main()
