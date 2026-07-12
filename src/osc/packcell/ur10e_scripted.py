"""Actuator-only physical upper bound for light_sortable_v1."""
from __future__ import annotations
import itertools
import numpy as np
import mujoco
from scipy.optimize import least_squares
from osc.robot_api import ActuatorCommand

R_DOWN=np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])

class UR10eScriptedController:
    def __init__(self,robot): self.robot=robot; self.m=robot.model; self.site=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site'); self.trace=[]
    def solve(self,target,start):
        d=mujoco.MjData(self.m);target=np.asarray(target)
        def residual(q):
            d.qpos[:6]=q;d.qpos[6:8]=[.075,-.075];mujoco.mj_forward(self.m,d);R=d.site_xmat[self.site].reshape(3,3);return np.r_[10*(d.site_xpos[self.site]-target),(R-R_DOWN).ravel()]
        starts=[np.asarray(start),np.array([2.8,-1.12,2.05,.65,1.57,1.23]),np.array([-2.8,-1.12,2.05,.65,-1.57,-1.23])]
        results=[least_squares(residual,s,bounds=(-np.pi*np.ones(6),np.pi*np.ones(6)),max_nfev=700) for s in starts];exact=[r for r in results if np.linalg.norm(r.fun)<1e-4];best=min(exact,key=lambda r:np.linalg.norm(r.x-np.asarray(start))) if exact else min(results,key=lambda r:np.linalg.norm(r.fun));return best.x,float(np.linalg.norm(residual(best.x)[:3])/10)
    def move(self,q,grip,phase,steps=180,frame_cb=None):
        start=np.asarray(self.robot.observe().joint_position)
        for i in range(steps):
            u=(i+1)/steps;u=u*u*(3-2*u);target=start+(np.asarray(q)-start)*u
            obs=self.robot.step(ActuatorCommand(tuple(target),tuple(grip),phase));
            if frame_cb and i%2==0:frame_cb(phase)
            object_position=self.robot.scorer_state()['object_position'];self.trace.append({'phase':phase,'joint_error':float(np.linalg.norm(np.asarray(obs.joint_position)-q)),'gripper':list(obs.gripper_position),'forces_n':list(obs.contact_forces_n),'object_position':object_position.tolist(),'object_z':float(object_position[2])})
    def close_guarded(self,q,frame_cb=None):
        held=None; bilateral=0;left_target=.075;right_target=-.075
        left_pad=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'left_pad');right_pad=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'right_pad')
        for _ in range(1000):
            obs=self.robot.step(ActuatorCommand(tuple(q),(float(left_target),float(right_target)),'close_guarded'))
            contacts={}
            for i,c in enumerate(self.robot.data.contact):
                pair={int(c.geom1),int(c.geom2)}
                if self.robot._parcel in pair and pair&self.robot._pads:
                    pad=next(iter(pair&self.robot._pads));f=np.zeros(6);mujoco.mj_contactForce(self.m,self.robot.data,i,f);contacts[pad]=abs(float(f[0]))
            bilateral=bilateral+1 if len(contacts)==2 and min(contacts.values())>=1.0 and max(contacts.values())<=125 else 0
            if frame_cb and len(self.trace)%2==0:frame_cb('close_guarded')
            self.trace.append({'phase':'close_guarded','requested_targets_m':[left_target,right_target],'contacts':contacts,'object_z':float(self.robot.scorer_state()['object_position'][2])})
            if bilateral>=5:
                # Add a small force-command preload after physical bilateral
                # contact; actuator force remains capped at 62.5 N per jaw.
                held=(float(max(.005,obs.gripper_position[0]-.006)),float(min(-.005,obs.gripper_position[1]+.006)));break
            if left_pad not in contacts:left_target=max(.005,left_target-.0001)
            if right_pad not in contacts:right_target=min(-.005,right_target+.0001)
        return held
    def cartesian_path(self,start_xyz,end_xyz,grip,phase,segments=12,steps_per_segment=100,frame_cb=None):
        q=np.asarray(self.robot.observe().joint_position);err=0.
        for point in np.linspace(np.asarray(start_xyz),np.asarray(end_xyz),segments+1)[1:]:
            q,err=self.solve(point,q);self.move(q,grip,phase,steps_per_segment,frame_cb)
        return q,err
    def run(self,frame_cb=None):
        self.robot.reset();home=np.asarray(self.robot.observe().joint_position);open_grip=(.075,-.075)
        poses={};q=home
        for name,target in [('pregrasp',(.52,0,.93)),('insertion',(.52,0,.78)),('lift',(.52,0,.96)),('transport',(.34,-.16,.96)),('lower',(.34,-.16,.82)),('retreat',(.34,-.16,.96))]:
            q,err=self.solve(target,q);poses[name]={'q':q,'error_m':err}
        self.move(poses['pregrasp']['q'],open_grip,'pregrasp',500,frame_cb);self.move(poses['insertion']['q'],open_grip,'insertion',400,frame_cb);held=self.close_guarded(poses['insertion']['q'],frame_cb)
        if held is None:return {'success':False,'failure_phase':'contact','poses':{k:{'error_m':v['error_m']} for k,v in poses.items()},'trace':self.trace}
        hold_q=np.asarray(self.robot.observe().joint_position);self.move(hold_q,held,'grasp_dwell',200,frame_cb)
        lift_q,lift_err=self.cartesian_path((.52,0,.78),(.52,0,.96),held,'lift',18,70,frame_cb);poses['lift']={'q':lift_q,'error_m':lift_err};retained=self.robot.scorer_state()['object_position'][2]>.83
        if not retained:return {'success':False,'failure_phase':'lift_retention','poses':{k:{'error_m':v['error_m']} for k,v in poses.items()},'trace':self.trace}
        transport_q,transport_err=self.cartesian_path((.52,0,.96),(.34,-.16,.96),held,'transport',16,80,frame_cb);poses['transport']={'q':transport_q,'error_m':transport_err}
        lower_q,lower_err=self.cartesian_path((.34,-.16,.96),(.34,-.16,.82),held,'lower',14,50,frame_cb);poses['lower']={'q':lower_q,'error_m':lower_err};self.move(np.asarray(self.robot.observe().joint_position),open_grip,'release',160,frame_cb)
        retreat_q,retreat_err=self.solve((.34,-.16,.96),np.asarray(self.robot.observe().joint_position));poses['retreat']={'q':retreat_q,'error_m':retreat_err};self.move(retreat_q,open_grip,'retreat',500,frame_cb)
        return {'success':bool(self.robot.verify()['camera_estimate_inside_box'] and self.robot.scorer_state()['inside_box']),'failure_phase':None,'poses':{k:{'error_m':v['error_m']} for k,v in poses.items()},'camera_verification':self.robot.verify(),'scorer_verification':self.robot.scorer_state()['inside_box'],'object_pose_writes_after_reset':False,'trace':self.trace}
