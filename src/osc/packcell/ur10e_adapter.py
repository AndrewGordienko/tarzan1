"""MuJoCo UR10e + simplified 2F-140 adapter for light_sortable_v1."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco
from osc.robot_api import ActuatorCommand, RobotArmAPI, RobotObservation

ROOT=Path(__file__).resolve().parents[3]; MODEL=ROOT/'assets/industrial/derived/ur10e_2f140_proxy/ur10e_2f140.xml'; SCENE=ROOT/'assets/industrial/derived/ur10e_2f140_proxy/light_sortable_scene.xml'

def build_scene() -> Path:
    xml=MODEL.read_text().replace('<worldbody>','<visual><global offwidth="1280" offheight="720"/></visual><worldbody>')
    env='''<geom name="table" type="box" pos="0.45 0 0.72" size="0.55 0.48 0.03" rgba="0.65 0.67 0.68 1"/>
    <body name="parcel" pos="0.52 0 0.78"><freejoint name="parcel_free"/><geom name="parcel" type="box" size="0.025 0.025 0.025" mass="0.1" friction="0.8 0.01 0.001" rgba="0.18 0.48 0.78 1"/></body>
    <body name="carton" pos="0.34 -0.16 0.77"><geom name="box_floor" type="box" size="0.16 0.12 0.01" rgba="0.58 0.37 0.18 1"/><geom name="box_back" type="box" pos="0 0.12 0.10" size="0.16 0.01 0.10" rgba="0.58 0.37 0.18 1"/><geom name="box_left" type="box" pos="-0.16 0 0.10" size="0.01 0.12 0.10" rgba="0.58 0.37 0.18 1"/><geom name="box_right" type="box" pos="0.16 0 0.10" size="0.01 0.12 0.10" rgba="0.58 0.37 0.18 1"/></body>
    <camera name="front_camera" pos="1.45 -1.35 1.35" xyaxes="0.70 0.71 0 -0.35 0.35 0.87"/><camera name="overhead_camera" pos="0.45 0 1.85" xyaxes="1 0 0 0 1 0"/>'''
    SCENE.write_text(xml.replace('</worldbody>',env+'</worldbody>')); return SCENE

class UR10eAdapter(RobotArmAPI):
    def __init__(self):
        build_scene(); self.model=mujoco.MjModel.from_xml_path(str(SCENE)); self.data=mujoco.MjData(self.model); self._object_qadr=int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')]); self._parcel=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'parcel'); self._pads={mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('left_pad','right_pad')}; self._post_reset=False
    def reset(self,seed=0):
        mujoco.mj_resetData(self.model,self.data); self.data.qpos[:6]=[2.8,-1.12,2.05,.65,1.57,1.23]; self.data.qpos[6:8]=[.075,-.075]; self.data.ctrl[:6]=self.data.qpos[:6]; self.data.ctrl[6:8]=[.075,-.075]; self.data.qpos[self._object_qadr:self._object_qadr+3]=[.52,0,.78]; self.data.qpos[self._object_qadr+3:self._object_qadr+7]=[1,0,0,0]; self.data.qvel[:]=0; mujoco.mj_forward(self.model,self.data); self._post_reset=True; return self.observe()
    def observe(self):
        forces=[]
        for i,c in enumerate(self.data.contact):
            if self._parcel in {int(c.geom1),int(c.geom2)} and ({int(c.geom1),int(c.geom2)}&self._pads):
                f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);forces.append(abs(float(f[0])))
        return RobotObservation(tuple(self.data.qpos[:6]),tuple(self.data.qvel[:6]),tuple(self.data.qpos[6:8]),tuple(forces),{})
    def step(self,command):
        self.data.ctrl[:6]=command.joint_targets; self.data.ctrl[6:8]=command.gripper_targets; mujoco.mj_step(self.model,self.data); return self.observe()
    def verify(self):
        p=np.asarray(self.data.geom_xpos[self._parcel]); inside=abs(p[0]-.34)<.15 and abs(p[1]+.16)<.11 and .77<p[2]<1.0
        return {'camera_estimate_inside_box':bool(inside),'released':not bool(self.observe().contact_forces_n)}
    def scorer_state(self):
        p=np.asarray(self.data.geom_xpos[self._parcel]);v=self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')]:][:6]
        return {'object_position':p.copy(),'object_velocity':v.copy(),'inside_box':bool(abs(p[0]-.34)<.15 and abs(p[1]+.16)<.11 and .77<p[2]<1.0)}
    def site_position(self): return np.asarray(self.data.site_xpos[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_site')]).copy()
