from __future__ import annotations
import argparse,json
from .packcell.reachability import run_reachability
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/packcell_reachability.json'); a=ap.parse_args(); r=run_reachability(); open(a.out,'w').write(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
