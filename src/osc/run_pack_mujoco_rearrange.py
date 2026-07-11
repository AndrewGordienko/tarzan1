from __future__ import annotations

import argparse
import json

from .embodied.rearrangement import run_rearrangement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/embodied_rearrangement_dev_100_109.json")
    args = ap.parse_args()
    print(json.dumps(run_rearrangement(out_path=args.out), indent=2))


if __name__ == "__main__": main()
