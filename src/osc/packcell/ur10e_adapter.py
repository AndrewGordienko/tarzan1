"""MuJoCo UR10e + simplified 2F-140 adapter for light_sortable_v1."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco
from osc.robot_api import ActuatorCommand, RobotArmAPI, RobotObservation

ROOT=Path(__file__).resolve().parents[3]; MODEL=ROOT/'assets/industrial/derived/ur10e_2f140_proxy/ur10e_2f140.xml'; SCENE=ROOT/'assets/industrial/derived/ur10e_2f140_proxy/light_sortable_scene.xml'
PICK=(.52,0.,.78); GRASP_INSERTION=(.52,0.,.800); CARTON=(.34,-.32,.77); CARTON_INTERIOR=(.32,.24,.20)

def build_scene() -> Path:
    xml=MODEL.read_text().replace('<worldbody>','<visual><global offwidth="1280" offheight="720"/></visual><worldbody>')
    env='''<geom name="table" type="box" pos="0.60 0 0.72" size="0.40 0.48 0.03" rgba="0.65 0.67 0.68 1"/>
    <body name="parcel" pos="0.52 0 0.78"><freejoint name="parcel_free"/><geom name="parcel" type="box" size="0.025 0.025 0.025" mass="0.1" friction="0.8 0.01 0.001" rgba="0.18 0.48 0.78 1"/></body>
    <body name="carton" pos="0.34 -0.32 0.77"><geom name="box_floor" type="box" size="0.16 0.12 0.01" rgba="0.58 0.37 0.18 1"/><geom name="box_back" type="box" pos="0 0.12 0.10" size="0.16 0.01 0.10" rgba="0.58 0.37 0.18 1"/><geom name="box_left" type="box" pos="-0.16 0 0.10" size="0.01 0.12 0.10" rgba="0.58 0.37 0.18 1"/><geom name="box_right" type="box" pos="0.16 0 0.10" size="0.01 0.12 0.10" rgba="0.58 0.37 0.18 1"/></body>
    <camera name="front_camera" pos="1.45 -1.35 1.35" xyaxes="0.70 0.71 0 -0.35 0.35 0.87"/><camera name="overhead_camera" pos="0.45 0 1.85" xyaxes="1 0 0 0 1 0"/>'''
    SCENE.write_text(xml.replace('</worldbody>',env+'</worldbody>')); return SCENE

class UR10eAdapter(RobotArmAPI):
    def __init__(self,support_lip=True,telemetry=False,object_mass_kg=.1,gravity_enabled=True):
        build_scene(); self.model=mujoco.MjModel.from_xml_path(str(SCENE)); self.data=mujoco.MjData(self.model); self._object_qadr=int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')]); self._parcel=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'parcel'); self._pads={mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('left_pad','right_pad')}; self._lips={mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('left_support_lip','right_support_lip')}; self._post_reset=False;self.telemetry_enabled=telemetry;self.telemetry=[];self._last_tcp=self._last_velocity=None;self._last_relative=None
        parcel_body=int(self.model.geom_bodyid[self._parcel]);scale=float(object_mass_kg/self.model.body_mass[parcel_body]);self.model.body_mass[parcel_body]*=scale;self.model.body_inertia[parcel_body]*=scale;self.object_mass_kg=float(object_mass_kg);self.model.opt.gravity[:]=[0,0,-9.81 if gravity_enabled else 0.];self.gravity_enabled=bool(gravity_enabled)
        if not support_lip:
            for g in self._lips:self.model.geom_contype[g]=0;self.model.geom_conaffinity[g]=0
    def reset(self,seed=0):
        mujoco.mj_resetData(self.model,self.data); self.data.qpos[:6]=[2.8,-1.12,2.05,.65,1.57,1.23]; self.data.qpos[6:8]=[.075,-.075]; self.data.ctrl[:6]=self.data.qpos[:6]; self.data.ctrl[6:8]=[.075,-.075]; self.data.qpos[self._object_qadr:self._object_qadr+3]=[.52,0,.78]; self.data.qpos[self._object_qadr+3:self._object_qadr+7]=[1,0,0,0]; self.data.qvel[:]=0; mujoco.mj_forward(self.model,self.data); self._post_reset=True;self.telemetry=[];self._last_tcp=self._last_velocity=self._last_relative=None; return self.observe()
    def observe(self):
        forces=[]
        for i,c in enumerate(self.data.contact):
            if self._parcel in {int(c.geom1),int(c.geom2)} and ({int(c.geom1),int(c.geom2)}&self._pads):
                f=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,f);forces.append(abs(float(f[0])))
        return RobotObservation(tuple(self.data.qpos[:6]),tuple(self.data.qvel[:6]),tuple(self.data.qpos[6:8]),tuple(forces),{})
    def tool_contact_metrics(self,pads_only=False):
        allowed=self._pads if pads_only else self._pads|self._lips
        sides={'left_jaw':{'normal_n':0.,'tangential_n':0.,'contacts':0},'right_jaw':{'normal_n':0.,'tangential_n':0.,'contacts':0}}
        distances=[]
        for i,c in enumerate(self.data.contact):
            pair={int(c.geom1),int(c.geom2)};tool=pair&allowed
            if self._parcel not in pair or not tool:continue
            force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,force);distances.append(float(c.dist))
            for geom in tool:
                side=self.model.body(int(self.model.geom_bodyid[geom])).name
                if side in sides:
                    sides[side]['normal_n']+=abs(float(force[0]));sides[side]['tangential_n']+=float(np.linalg.norm(force[1:3]));sides[side]['contacts']+=1
        return {'left':sides['left_jaw'],'right':sides['right_jaw'],'minimum_distance_m':min(distances) if distances else None,'bilateral':sides['left_jaw']['contacts']>0 and sides['right_jaw']['contacts']>0}
    def effective_pad_friction(self):
        return min(float(self.model.geom_friction[g,0]) for g in self._pads|{self._parcel})
    def parcel_relative_to_grasp_site(self):
        sid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_site');R=np.asarray(self.data.site_xmat[sid]).reshape(3,3);return R.T@(np.asarray(self.data.geom_xpos[self._parcel])-np.asarray(self.data.site_xpos[sid]))
    def step(self,command):
        self.data.ctrl[:6]=command.joint_targets; self.data.ctrl[6:8]=command.gripper_targets; mujoco.mj_step(self.model,self.data)
        if self.telemetry_enabled:self._capture(command)
        return self.observe()
    def _capture(self,command):
        dt=float(self.model.opt.timestep);sid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_site');tcp=np.asarray(self.data.site_xpos[sid]).copy();R=np.asarray(self.data.site_xmat[sid]).reshape(3,3);obj=np.asarray(self.data.geom_xpos[self._parcel]);relative=R.T@(obj-tcp);velocity=np.zeros(3) if self._last_tcp is None else (tcp-self._last_tcp)/dt;accel=np.zeros(3) if self._last_velocity is None else (velocity-self._last_velocity)/dt;jerk=np.zeros(3) if len(self.telemetry)<2 else (accel-np.asarray(self.telemetry[-1]['tcp_acceleration_mps2']))/dt;slip=np.zeros(3) if self._last_relative is None else (relative-self._last_relative)/dt
        contacts=[];prohibited_impulse=0.;tool_contacts=self._pads|self._lips
        robot_bodies={
            'base','shoulder_link','upper_arm_link','forearm_link','wrist_1_link',
            'wrist_2_link','wrist_3_link','tool_body','left_jaw','right_jaw',
        }
        for i,c in enumerate(self.data.contact):
            force=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,force);pair={int(c.geom1),int(c.geom2)};normal=abs(float(force[0]));tangent=float(np.linalg.norm(force[1:3]));mu=float(min(self.model.geom_friction[int(c.geom1),0],self.model.geom_friction[int(c.geom2),0]));kind='tool_parcel' if self._parcel in pair and pair&tool_contacts else 'other';impulse=normal*dt
            body_names={self.model.body(int(self.model.geom_bodyid[g])).name for g in pair}
            robot_involved=body_names&robot_bodies
            if kind=='tool_parcel': collision_class='intended_tool_parcel'
            elif len(robot_involved)>1: collision_class='robot_self'
            elif robot_involved and 'carton' in body_names: collision_class='robot_box'
            elif robot_involved and 'world' in body_names: collision_class='robot_table'
            elif body_names=={'parcel','world'}: collision_class='parcel_table_support'
            elif body_names=={'carton','world'}: collision_class='box_table_support'
            else: collision_class='other'
            prohibited=collision_class in {'robot_self','robot_box','robot_table'}
            if prohibited:prohibited_impulse+=impulse
            contacts.append({'geom1':self.model.geom(int(c.geom1)).name,'geom2':self.model.geom(int(c.geom2)).name,'body1':self.model.body(int(self.model.geom_bodyid[int(c.geom1)])).name,'body2':self.model.body(int(self.model.geom_bodyid[int(c.geom2)])).name,'kind':kind,'collision_class':collision_class,'prohibited':prohibited,'normal_n':normal,'tangential_n':tangent,'friction_cone_utilization':tangent/max(mu*normal,1e-9),'dist_m':float(c.dist),'impulse_ns':impulse})
        jaw_dofs=[int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in ('left_jaw_joint','right_jaw_joint')]
        pad_metrics=self.tool_contact_metrics(pads_only=True);tool_metrics=self.tool_contact_metrics(pads_only=False)
        self.telemetry.append({'step':len(self.telemetry),'time_s':float(self.data.time),'controller_state':command.phase,'gripper_writer':'UR10eAdapter.step/ActuatorCommand','gripper_command_m':list(command.gripper_targets),'jaw_position_m':self.data.qpos[6:8].tolist(),'jaw_aperture_m':float(self.data.qpos[6]-self.data.qpos[7]-.01),'jaw_velocity_mps':self.data.qvel[jaw_dofs].tolist(),'jaw_actuator_force_n':self.data.qfrc_actuator[jaw_dofs].tolist(),'pad_force_left_n':pad_metrics['left'],'pad_force_right_n':pad_metrics['right'],'tool_force_left_n':tool_metrics['left'],'tool_force_right_n':tool_metrics['right'],'parcel_position_m':obj.tolist(),'parcel_relative_to_gripper_m':relative.tolist(),'linear_slip_mps':slip.tolist(),'tcp_velocity_mps':velocity.tolist(),'tcp_acceleration_mps2':accel.tolist(),'tcp_jerk_mps3':jerk.tolist(),'angular_velocity_rps':self.data.cvel[self.model.site_bodyid[sid],:3].tolist(),'contacts':contacts,'collision_impulse_ns':prohibited_impulse})
        self._last_tcp=tcp;self._last_velocity=velocity;self._last_relative=relative
    def verify(self):
        p=np.asarray(self.data.geom_xpos[self._parcel]); inside=abs(p[0]-CARTON[0])<.15 and abs(p[1]-CARTON[1])<.11 and .77<p[2]<1.0
        return {'camera_estimate_inside_box':bool(inside),'released':not bool(self.observe().contact_forces_n)}
    def scorer_state(self):
        p=np.asarray(self.data.geom_xpos[self._parcel]);v=self.data.qvel[self.model.jnt_dofadr[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'parcel_free')]:][:6]
        return {'object_position':p.copy(),'object_velocity':v.copy(),'inside_box':bool(abs(p[0]-CARTON[0])<.15 and abs(p[1]-CARTON[1])<.11 and .77<p[2]<1.0)}
    def site_position(self): return np.asarray(self.data.site_xpos[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_site')]).copy()
