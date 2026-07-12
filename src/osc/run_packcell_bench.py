from __future__ import annotations
import argparse, json
from .packcell.physical import run_packcell_benchmark

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=20); args=ap.parse_args()
    print(json.dumps(run_packcell_benchmark(range(args.episodes)), indent=2))
if __name__=='__main__': main()
