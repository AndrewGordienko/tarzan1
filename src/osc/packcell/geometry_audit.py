from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .physical import PackCell


def _geom_record(cell, gid):
    m=cell.model; g=m.geom(gid)
    return {"id":int(gid),"name":g.name or f"geom_{gid}","body":m.body(m.geom_bodyid[gid]).name,
            "type":int(m.geom_type[gid]),"size":m.geom_size[gid].tolist(),
            "contype":int(m.geom_contype[gid]),"conaffinity":int(m.geom_conaffinity[gid]),
            "friction":m.geom_friction[gid].tolist(),"world_position":cell.data.geom_xpos[gid].tolist()}


def audit(seed=0):
    c=PackCell(seed); c.reset(); m,d=c.model,c.data; obj=m.geom("cube_red").id
    # The calibrated asset has a fixed gripper collision mesh and a moving jaw.
    gripper=m.body("gripper").id
    candidates=[i for i in range(m.ngeom) if m.geom_contype[i]>0 and m.geom_bodyid[i] in {gripper, m.body("moving_jaw_so101_v1").id}]
    fixed=next(i for i in candidates if i != m.geom("cube_red").id and m.geom_bodyid[i] == gripper and i >= 29)
    moving=next(i for i in candidates if m.geom_bodyid[i] != gripper)
    def distance(a,b):
        fromto=np.zeros(6); return float(mj.mj_geomDistance(m,d,int(a),int(b),1.0,fromto))
    import mujoco as mj
    start={"fixed_to_object":distance(fixed,obj),"moving_to_object":distance(moving,obj),
           "fingertip_separation":float(np.linalg.norm(d.geom_xpos[fixed]-d.geom_xpos[moving])),
           "object_center":d.geom_xpos[obj].tolist(),"object_size":(m.geom_size[obj]*2).tolist(),
           "grasp_center":((d.geom_xpos[fixed]+d.geom_xpos[moving])/2).tolist(),
           "fixed":_geom_record(c,fixed),"moving":_geom_record(c,moving),"object":_geom_record(c,obj)}
    # Close while holding pose; record the minimum geometric distances.
    mins={"fixed_to_object":start["fixed_to_object"],"moving_to_object":start["moving_to_object"],"fingertip_separation":start["fingertip_separation"]}
    for _ in range(100):
        d.ctrl[5]=-.17; mj.mj_step(m,d)
        mins["fixed_to_object"]=min(mins["fixed_to_object"],distance(fixed,obj)); mins["moving_to_object"]=min(mins["moving_to_object"],distance(moving,obj)); mins["fingertip_separation"]=min(mins["fingertip_separation"],float(np.linalg.norm(d.geom_xpos[fixed]-d.geom_xpos[moving])))
    return {"seed":seed,"start":start,"minimum_during_close":mins,"diagnosis":"both_distances_positive" if mins["fixed_to_object"]>0 and mins["moving_to_object"]>0 else "contact_configuration_or_asymmetry"}

if __name__=="__main__": print(json.dumps(audit(),indent=2))
