from __future__ import annotations
from pathlib import Path
import json,numpy as np,mujoco
from render_panda_motion_demo import make_scene,ROOT

def main():
 scene=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_aperture_sweep_scene.xml';make_scene(scene)
 try:
  m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'parcel');qobj=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')];qf=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in ('finger_joint1','finger_joint2')];d.qpos[:7]=[0,-.5,0,-2,0,1.5,.7];mujoco.mj_forward(m,d);d.qpos[qobj:qobj+3]=d.site_xpos[sid];mujoco.mj_forward(m,d);rows=[]
  for aperture in np.linspace(.04,0,41):
   d.qpos[qf]=aperture;d.qvel[:]=0;mujoco.mj_forward(m,d);dist=[float(mujoco.mj_geomDistance(m,d,i,obj,1.0,np.zeros(6))) for i in range(m.ngeom) if m.geom_bodyid[i] in {mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'left_finger'),mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_finger')} and m.geom_contype[i]>0]; rows.append({'aperture_joint_m':float(aperture),'finger_distances_m':dist,'fingertip_separation_m':float(abs(d.qpos[qf[0]]-d.qpos[qf[1]])),'symmetry_m':float(abs(d.qpos[qf[0]]-d.qpos[qf[1]])),'parcel_relative_to_grasp_site_m':(d.geom_xpos[obj]-d.site_xpos[sid]).tolist(),'contact_margin_m':min(dist),'solref':m.opt.solver,'solimp':m.geom_solimp[obj].tolist(),'timestep_s':m.opt.timestep,'actuator_limits':m.actuator_ctrlrange[7].tolist()})
  valid=[r for r in rows if max(r['finger_distances_m'])<=0 and min(r['finger_distances_m'])>=-.002];out={'dynamic_object':{'free_joint':True,'mass_kg':float(m.body_mass[m.geom_bodyid[obj]]),'mocap':False,'weld':False,'initialized_only_at_reset':True},'rows':rows,'valid_apertures':valid};(ROOT/'artifacts/panda_aperture_sweep.json').write_text(json.dumps(out,indent=2));print({'samples':len(rows),'valid_apertures':len(valid),'mass_kg':out['dynamic_object']['mass_kg']})
 finally:scene.unlink(missing_ok=True)
if __name__=='__main__':main()
