from __future__ import annotations
import argparse,json
from .packcell.layout_sweep import sweep
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/packcell_layout_sweep.json'); a=ap.parse_args(); r=sweep(); open(a.out,'w').write(json.dumps(r,indent=2)); print(json.dumps({"selected":r["selected"],"intervention_required":r["intervention_required"]},indent=2))
if __name__=='__main__': main()
