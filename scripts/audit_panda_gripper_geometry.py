from __future__ import annotations
from pathlib import Path
import json,numpy as np,mujoco
from render_panda_motion_demo import make_scene,ROOT

def run(centered=False):
    scene=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_gripper_audit_scene.xml'; make_scene(scene)
    try:
        m=mujoco.MjModel.from_xml_path(str(scene)); d=mujoco.MjData(m); sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site'); obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'parcel'); lb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_finger'); rb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_finger'); fingers=[i for i in range(m.ngeom) if m.geom_bodyid[i] in {lb,rb} and m.geom_contype[i]>0]; qobj=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')]; d.qpos[:7]=[0,-.5,0,-2,0,1.5,.7]; mujoco.mj_forward(m,d)
        if centered: d.qpos[qobj:qobj+3]=d.site_xpos[sid]; mujoco.mj_forward(m,d)
        initial=d.geom_xpos[obj].copy(); rows=[]
        for _ in range(120):
            d.ctrl[:7]=d.qpos[:7]; d.ctrl[7]=0; mujoco.mj_step(m,d); dist=[float(mujoco.mj_geomDistance(m,d,i,obj,1.0,np.zeros(6))) for i in fingers]; classes={'finger_parcel':[],'finger_finger':[],'robot_environment':[],'parcel_table':[]}
            for i,c in enumerate(d.contact):
                a,b=int(c.geom1),int(c.geom2); f=np.zeros(6); mujoco.mj_contactForce(m,d,i,f); pair=(m.geom(a).name or f'geom_{a}',m.geom(b).name or f'geom_{b}'); rec={'pair':pair,'normal_force_n':abs(float(f[0])),'penetration_m':max(0.,-float(c.dist))}
                if obj in {a,b} and bool({a,b}&set(fingers)): classes['finger_parcel'].append(rec)
                elif {a,b}.issubset(set(fingers)): classes['finger_finger'].append(rec)
                elif obj in {a,b} and m.geom('table').id in {a,b}: classes['parcel_table'].append(rec)
                else: classes['robot_environment'].append(rec)
            rows.append({'parcel_position':d.geom_xpos[obj].tolist(),'displacement_m':float(np.linalg.norm(d.geom_xpos[obj]-initial)),'finger_separation_m':float(np.linalg.norm(d.geom_xpos[fingers[0]]-d.geom_xpos[fingers[-1]])),'signed_distances_m':dist,'classes':classes,'finger_actuator_force':float(d.qfrc_actuator[7])})
        fp=[x for r in rows for x in r['classes']['finger_parcel']]
        return {'centered_reset':centered,'parcel_size_m':(m.geom_size[obj]*2).tolist(),'fingers':[{'id':i,'name':m.geom(i).name or f'geom_{i}','contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i]),'friction':m.geom_friction[i].tolist()} for i in fingers],'initial_signed_distances_m':rows[0]['signed_distances_m'],'rows':rows,'max_normal_force_n':max([x['normal_force_n'] for x in fp] or [0]),'min_signed_distance_m':min([x for r in rows for x in r['signed_distances_m']] or [0]),'valid_opposing_grasp':bool(centered and fp and max(x['penetration_m'] for x in fp)<=.002 and max(x['normal_force_n'] for x in fp)<=70)}
    finally: scene.unlink(missing_ok=True)
if __name__=='__main__':
 out={'empty_closure':run(False),'centered_closure':run(True)}; (ROOT/'artifacts/panda_gripper_geometry_audit.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:{'max_force_n':v['max_normal_force_n'],'min_distance_m':v['min_signed_distance_m'],'valid':v['valid_opposing_grasp']} for k,v in out.items()},indent=2))
