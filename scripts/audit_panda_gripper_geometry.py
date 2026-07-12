from __future__ import annotations
from pathlib import Path
import tempfile,json,numpy as np,mujoco
from render_panda_motion_demo import make_scene,ROOT

def run(centered=False):
    scene=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_gripper_audit_scene.xml';make_scene(scene)
    try:
        m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'parcel');fingers=[i for i in range(m.ngeom) if m.geom_bodyid[i] in {mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_finger'),mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_finger')} and m.geom_contype[i]>0]; qobj=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')];d.qpos[:7]=[0,-.5,0,-2,0,1.5,.7];mujoco.mj_forward(m,d)
        if centered: d.qpos[qobj:qobj+3]=d.site_xpos[sid]; mujoco.mj_forward(m,d)
        rows=[]
        for _ in range(120):
            d.ctrl[:7]=d.qpos[:7];d.ctrl[7]=0;mujoco.mj_step(m,d); pairs=[];forces=[]; dist=[]
            for i in fingers:
                fromto=np.zeros(6);dist.append(float(mujoco.mj_geomDistance(m,d,i,obj,1.0,fromto)))
            for i,c in enumerate(d.contact):
                if obj in {c.geom1,c.geom2}:
                    f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);pairs.append((m.geom(c.geom1).name,m.geom(c.geom2).name));forces.append(abs(float(f[0])))
            rows.append({'finger_positions':[d.geom_xpos[i].tolist() for i in fingers],'parcel_position':d.geom_xpos[obj].tolist(),'finger_separation_m':float(np.linalg.norm(d.geom_xpos[fingers[0]]-d.geom_xpos[fingers[-1]])),'signed_distances_m':dist,'pairs':pairs,'normal_forces_n':forces,'finger_commands':d.ctrl[:7].tolist(),'finger_actuator_force':float(d.qfrc_actuator[7])})
        return {'centered_reset':centered,'parcel_size_m':(m.geom_size[obj]*2).tolist(),'fingers':[{'id':i,'name':m.geom(i).name or f'geom_{i}','contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i]),'friction':m.geom_friction[i].tolist()} for i in fingers],'rows':rows,'max_normal_force_n':max([f for r in rows for f in r['normal_forces_n']] or [0]),'min_signed_distance_m':min([x for r in rows for x in r['signed_distances_m']] or [0])}
    finally: scene.unlink(missing_ok=True)

if __name__=='__main__':
 out={'empty_closure':run(False),'centered_closure':run(True)};Path=ROOT/'artifacts/panda_gripper_geometry_audit.json';Path.write_text(json.dumps(out,indent=2));print(json.dumps({k:{'max_force_n':v['max_normal_force_n'],'min_distance_m':v['min_signed_distance_m']} for k,v in out.items()},indent=2))
