from __future__ import annotations
import argparse,json
from .packcell.attribution import run
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/packcell_attribution_ladder.json'); a=ap.parse_args(); print(json.dumps(run(a.out),indent=2))
if __name__=='__main__': main()
