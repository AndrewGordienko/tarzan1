from __future__ import annotations
from pathlib import Path
import tempfile, numpy as np, mujoco, json
from render_panda_motion_demo import make_scene, ROOT

def main():
    scene=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_grasp_lift_scene.xml'; make_scene(scene)
    try:
        m=mujoco.MjModel.from_xml_path(str(scene)); d=mujoco.MjData(m); sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site'); qj=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}')] for i in range(1,8)]; dj=[m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}')] for i in range(1,8)]; obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'parcel'); left=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_finger'); right=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_finger'); finger={i for i in range(m.ngeom) if m.geom_bodyid[i] in {left,right}}
        d.qpos[:7]=[0,-.5,0,-2,0,1.5,.7]; mujoco.mj_forward(m,d); phases=[('pregrasp',(.55,0,.30),255,120),('insertion',(.55,0,.18),255,100),('closure',(.55,0,.18),0,100),('lift',(.55,0,.35),0,120)]; logs=[]
        for name,target,grip,n in phases:
            for _ in range(n):
                J=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,J,None,sid); JJ=J[:,dj]; err=np.asarray(target)-d.site_xpos[sid]; dq=JJ.T@np.linalg.solve(JJ@JJ.T+.05**2*np.eye(3),err)*.7; q=d.qpos[qj].copy()+np.clip(dq,-.04,.04); d.ctrl[:7]=q; d.ctrl[7]=grip; mujoco.mj_step(m,d)
                pairs=[]; forces=[]
                for i,c in enumerate(d.contact):
                    if obj in {c.geom1,c.geom2} and ({c.geom1,c.geom2}&finger):
                        f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f); pairs.append((m.geom(c.geom1).name,m.geom(c.geom2).name)); forces.append(abs(float(f[0])))
                logs.append({'phase':name,'object_z':float(d.geom_xpos[obj][2]),'pairs':pairs,'normal_forces':forces,'site_error_m':float(np.linalg.norm(np.asarray(target)-d.site_xpos[sid]))})
        out={'phases':{p:{'steps':sum(x['phase']==p for x in logs),'max_force_n':max([f for x in logs if x['phase']==p for f in x['normal_forces']] or [0]),'max_object_z_m':max(x['object_z'] for x in logs if x['phase']==p)} for p,_,_,_ in phases},'object_pose_writes_after_reset':False,'logs':logs}
        print(json.dumps(out,indent=2)); (ROOT/'artifacts/panda_grasp_lift_diagnostics.json').write_text(json.dumps(out,indent=2))
    finally: scene.unlink(missing_ok=True)
if __name__=='__main__': main()
