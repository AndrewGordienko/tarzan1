from __future__ import annotations
import itertools, json
from .physical import PackCell
from .candidate_search import search

def sweep():
    layouts=[]
    for h,yaw,x,y in itertools.product((0,.05,.10,.15),(-.25,0,.25),(.18,.22,.26),(-.06,0,.06)):
        layout={"base_height":h,"base_yaw":yaw,"pick_zone":[x,y],"box_sector":"opposite"}
        try:
            # Run the same bounded solver against this physical mount variant.
            result=search(seed=0, layout=layout, fast=True); valid=result["valid_count"]
            layouts.append({"layout":layout,"valid_candidates":valid,"robust_region_score":valid,"prohibited_links":result["prohibited_link_frequency"],"status":"candidate_search_only"})
        except Exception as exc: layouts.append({"layout":layout,"valid_candidates":0,"status":"error","error":str(exc)})
    best=max(layouts,key=lambda x:x["valid_candidates"])
    return {"baseline":{"layout":{"base_height":0,"base_yaw":0,"pick_zone":[.2,-.06],"box_sector":"same"},"status":"invalid_reference"},"layouts":layouts,"selected":best,"intervention_required":best["valid_candidates"]==0}
