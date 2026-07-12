from __future__ import annotations
import argparse,json
from .packcell.geometry_audit import audit

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='artifacts/packcell_geometry_audit.json'); a=ap.parse_args(); r=audit(); open(a.out,'w').write(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
