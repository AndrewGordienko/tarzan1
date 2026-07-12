from __future__ import annotations
import argparse,json
from .packcell.grasp_isolation import run_grasp_isolation

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=20); ap.add_argument('--out',default='artifacts/packcell_grasp_isolation.json'); a=ap.parse_args()
    report=run_grasp_isolation(range(a.episodes)); open(a.out,'w').write(json.dumps(report,indent=2)); print(json.dumps(report['modes'],indent=2))
if __name__=='__main__': main()
