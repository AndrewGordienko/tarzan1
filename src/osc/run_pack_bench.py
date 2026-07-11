"""Run the packing PoC benchmark."""
from __future__ import annotations

import argparse
import json
from .packing.benchmark import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--perception", choices=("oracle", "belief"), default="oracle")
    ap.add_argument("--out", default="artifacts/packing_report.json")
    ap.add_argument("--render-dir", default="artifacts/packing")
    args = ap.parse_args()
    report = run_benchmark(args.episodes, args.perception, args.out, args.render_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__": main()
