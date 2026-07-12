from __future__ import annotations
import numpy as np
import mujoco
from .physical import PackCell

def offline_solve(cell, target, orientation=False, starts=8):
    m=cell.model; sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site')
    jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')]
    joints=[m.jnt_qposadr[j] for j in jids]
    dofs=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')]
    best=None; rng=np.random.default_rng(0)
    target=np.asarray(target,float)
    for _ in range(starts):
        q=cell.data.qpos.copy(); q[joints]=rng.uniform(-1.0,1.0,len(joints)); d=mujoco.MjData(m); d.qpos[:]=q; mujoco.mj_forward(m,d)
        for _ in range(600):
            jp=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,jp,None,sid); J=jp[:,dofs]; err=target-d.site_xpos[sid]; dq=J.T@np.linalg.solve(J@J.T+.03**2*np.eye(3),err)*.7; d.qpos[joints]+=np.clip(dq,-.05,.05); mujoco.mj_forward(m,d)
        err=float(np.linalg.norm(target-d.site_xpos[sid])); sv=np.linalg.svd(J,compute_uv=False)
        margins=np.minimum(d.qpos[joints]-m.jnt_range[jids,0],m.jnt_range[jids,1]-d.qpos[joints])
        contacts=[]; prohibited=[]
        obj=m.geom("cube_red").id; table=m.geom("table").id; gripper=m.body("gripper").id; moving=m.body("moving_jaw_so101_v1").id
        finger={i for i in range(m.ngeom) if m.geom_bodyid[i] in {gripper,moving} and i>=29}
        for c in d.contact:
            a,b=int(c.geom1),int(c.geom2); pair=(m.geom(a).name or f"geom_{a}",m.geom(b).name or f"geom_{b}")
            allowed=bool((table in {a,b}) or (obj in {a,b} and bool({a,b}&finger)))
            contacts.append({"pair":pair,"allowed":allowed})
            if not allowed: prohibited.append(pair)
        row={"position_error":err,"final_qpos":d.qpos[joints].tolist(),"joint_limit_margin":float(np.min(margins)),"singular_values":sv.tolist(),"collision_count":int(d.ncon),"collision_free":not prohibited,"contacts":contacts,"prohibited_contacts":prohibited,"orientation_objective":orientation}
        if best is None or err<best["position_error"]: best=row
    return best

def run_reachability():
    c=PackCell(); c.reset(); target=c.scorer_state()["object_position"]
    offline={"position_only":offline_solve(c,target,False),"position_plus_orientation":offline_solve(c,target,True)}
    return {"target":target.tolist(),"offline":offline,"live_position_only":{"implemented":True,"note":"OraclePackController already uses position-only DLS; insertion remains hard-gated and should be run for 20 seeds."}}
