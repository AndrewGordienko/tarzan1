from __future__ import annotations
import argparse, json
from .packcell.physical import run_packcell_benchmark
from .packcell.physical import PackCell
from .packcell.controller import OraclePackController

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=20); args=ap.parse_args()
    rows=[]
    for seed in range(args.episodes):
        try:
            cell=PackCell(seed); cell.reset(); result=OraclePackController(cell, cell.scorer_state()["object_position"]).run()
            rows.append({"seed":seed,**result,"failure_attribution":None if result["success"] else "grasp"})
        except Exception as exc:
            rows.append({"seed":seed,"success":False,"failure_attribution":"environment","reason":str(exc)})
    print(json.dumps({"episodes":len(rows),"successes":sum(r.get("success",False) for r in rows),"rows":rows}, indent=2))
if __name__=='__main__': main()
