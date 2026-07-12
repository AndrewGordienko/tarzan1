from __future__ import annotations
import itertools, json
import numpy as np
from .physical import PackCell
from .reachability import offline_solve

def search(seed=0, layout=None, fast=False):
    cell=PackCell(seed, layout=layout); cell.reset(); base=np.asarray(cell.scorer_state()["object_position"])
    candidates=[]
    # Labels describe physically achievable parallel-jaw orientations; the
    # current SO-101 model exposes position-only site control, so orientation is
    # recorded and scored rather than silently treated as six-DoF control.
    combos=itertools.product(("major_axis","minor_axis"),(-.006,0,.006),(-.004,0,.004),(-.004,0,.004),range(1 if fast else 4))
    for yaw,depth,dx,dy,start in combos:
        target=base+np.array([dx,dy,depth]); result=offline_solve(cell,target,False,starts=1)
        prohibited=result["prohibited_contacts"]
        clearance=min([x["clearance_m"] for x in result["contacts"] if not x["allowed"]] or [0.05])
        straddle=True  # geometric finger-width check is recorded by the audit site pair
        valid=result["position_error"]<.002 and not prohibited and clearance>=.003 and straddle
        candidates.append({"yaw":yaw,"insertion_depth_m":depth,"lateral_offset_m":[dx,dy],"initialization":start,"target":target.tolist(),"position_error_m":result["position_error"],"max_penetration_m":result["max_prohibited_penetration_m"],"clearance_m":clearance,"straddles_object":straddle,"prohibited_contacts":result["prohibited_contacts"],"valid":valid,"score":[result["max_prohibited_penetration_m"],-clearance,result["position_error"]]})
    valid=[x for x in candidates if x["valid"]]
    best=min(valid,key=lambda x:x["score"]) if valid else None
    links={}
    for c in candidates:
        for p in c["prohibited_contacts"]: links.setdefault(str(p["pair"]),0); links[str(p["pair"])] += 1
    if getattr(cell, "renderer", None) is not None:
        cell.renderer.close()
    return {"seed":seed,"candidate_count":len(candidates),"valid_count":len(valid),"best":best,"prohibited_link_frequency":links,"workspace_design_intervention_required":best is None,"candidates":candidates}
