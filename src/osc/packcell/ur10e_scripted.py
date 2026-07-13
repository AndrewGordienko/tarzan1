"""Actuator-only physical upper bound for light_sortable_v1."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import mujoco
from scipy.optimize import least_squares
from osc.robot_api import ActuatorCommand
from osc.packcell.ur10e_adapter import CARTON, GRASP_INSERTION
from osc.packcell.retention import RetentionBudget, calculate_retention_budget

R_DOWN=np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]])

@dataclass
class ForceHoldState:
    budget: RetentionBudget
    gripper_targets: tuple[float,float]
    initial_relative_pose: np.ndarray
    previous_relative_pose: np.ndarray
    integral_error: list[float]
    adaptive_force_per_side_n: float
    bilateral_steps: int=0
    unilateral_steps: int=0
    saturated_steps: int=0
    abort_reason: str|None=None

class UR10eScriptedController:
    def __init__(self,robot): self.robot=robot; self.m=robot.model; self.site=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site'); self.trace=[];self.last_close_failure=None
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
        self.last_close_failure=None;held=None; bilateral=0;left_target=.075;right_target=-.075
        left_pad=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'left_pad');right_pad=mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_GEOM,'right_pad')
        width=2.*float(self.m.geom_size[self.robot._parcel,1]);precontact_aperture=width+.005;precontact_joint=(precontact_aperture+.01)/2.
        start=np.asarray(self.robot.observe().gripper_position)
        for i in range(3000):
            u=min(1.,(i+1)/500.);u=u*u*(3-2*u);grip=start+(np.asarray((precontact_joint,-precontact_joint))-start)*u
            self.robot.step(ActuatorCommand(tuple(q),tuple(grip),'precontact'))
            aperture=float(self.robot.data.qpos[6]-self.robot.data.qpos[7]-.01)
            if i>=500 and abs(aperture-precontact_aperture)<.0005 and max(abs(float(x)) for x in self.robot.data.qvel[6:8])<.002:break
        left_target,right_target=map(float,self.robot.observe().gripper_position)
        for _ in range(1500):
            obs=self.robot.step(ActuatorCommand(tuple(q),(float(left_target),float(right_target)),'close_guarded'))
            contacts={}
            for i,c in enumerate(self.robot.data.contact):
                pair={int(c.geom1),int(c.geom2)}
                if self.robot._parcel in pair and pair&(self.robot._pads|self.robot._lips):
                    tool=next(iter(pair&(self.robot._pads|self.robot._lips)));f=np.zeros(6);mujoco.mj_contactForce(self.m,self.robot.data,i,f);contacts[tool]=abs(float(f[0]))
            metrics=self.robot.tool_contact_metrics(pads_only=False);side_forces=[metrics['left']['normal_n'],metrics['right']['normal_n']]
            if metrics['minimum_distance_m'] is not None and metrics['minimum_distance_m']<-.002:self.last_close_failure='penetration_limit';return None
            bilateral=bilateral+1 if metrics['bilateral'] and min(side_forces)>=.05 and sum(side_forces)<=125 else max(0,bilateral-1)
            if frame_cb and len(self.trace)%2==0:frame_cb('close_guarded')
            self.trace.append({'phase':'close_guarded','requested_targets_m':[left_target,right_target],'contacts':contacts,'tool_normal_force_n':side_forces,'minimum_contact_distance_m':metrics['minimum_distance_m'],'object_z':float(self.robot.scorer_state()['object_position'][2])})
            if bilateral>=5:
                # Force preload is owned by the closed-loop retention controller.
                held=(float(obs.gripper_position[0]),float(obs.gripper_position[1]));break
            # Every closure target is derived from measured jaw position.  A
            # contacting jaw is actively braked instead of retaining stale
            # position error accumulated during free-space closure.
            left_creep=.00002 if metrics['left']['contacts'] else .00025
            right_creep=.00002 if metrics['right']['contacts'] else .00025
            left_target=max(.005,float(obs.gripper_position[0])-left_creep)
            right_target=min(-.005,float(obs.gripper_position[1])+right_creep)
        if held is None:self.last_close_failure='bilateral_contact_timeout'
        return held
    def start_force_hold(self,grip,budget):
        relative=self.robot.parcel_relative_to_grasp_site().copy()
        return ForceHoldState(budget,tuple(grip),relative,relative.copy(),[0.,0.],budget.target_normal_force_per_side_n)
    def force_hold_step(self,joint_targets,state,phase):
        if state.abort_reason:return False
        obs=self.robot.step(ActuatorCommand(tuple(joint_targets),state.gripper_targets,phase));metrics=self.robot.tool_contact_metrics(pads_only=False);pad_metrics=self.robot.tool_contact_metrics(pads_only=True)
        force=[metrics['left']['normal_n'],metrics['right']['normal_n']];total_force=sum(force);dt=float(self.m.opt.timestep);relative=self.robot.parcel_relative_to_grasp_site();slip_velocity=float(np.linalg.norm(relative-state.previous_relative_pose)/dt);drift=float(np.linalg.norm(relative-state.initial_relative_pose));state.previous_relative_pose=relative.copy()
        penetration=metrics['minimum_distance_m'];prohibited=bool(self.robot.telemetry_enabled and self.robot.telemetry and any(c['prohibited'] for c in self.robot.telemetry[-1]['contacts']))
        if slip_velocity>.003 and state.abort_reason is None:
            gain=.004 if slip_velocity>.02 else .001
            state.adaptive_force_per_side_n=min(state.budget.maximum_allowed_normal_force_n/2.,state.adaptive_force_per_side_n+gain)
        if penetration is not None and penetration<-.002:state.abort_reason='penetration_limit'
        elif total_force>state.budget.maximum_allowed_normal_force_n:state.abort_reason='force_limit'
        elif prohibited:state.abort_reason='prohibited_collision'
        elif drift>.020:state.abort_reason='persistent_slip'
        state.bilateral_steps=state.bilateral_steps+1 if metrics['bilateral'] else 0;state.unilateral_steps=0 if metrics['bilateral'] else state.unilateral_steps+1
        if state.unilateral_steps>100:state.abort_reason='bilateral_contact_lost'
        actual=list(obs.gripper_position);targets=list(state.gripper_targets);limits=[(.005,.075),(-.075,-.005)];actuator_kp=max(1.,float(self.m.actuator_gainprm[6,0]))
        for side in range(2):
            error=state.adaptive_force_per_side_n-force[side]
            saturated=(side==0 and targets[side]<=limits[side][0]+1e-8) or (side==1 and targets[side]>=limits[side][1]-1e-8)
            if not saturated and state.abort_reason is None:
                state.integral_error[side]=max(0.,min(20.,state.integral_error[side]+error*dt))
                # Position-actuator feed-forward plus bounded PI feedback.  The
                # target is derived from measured qpos each step, not a stale
                # desired aperture, so the loop cannot accumulate hidden close.
                desired_force=max(0.,state.adaptive_force_per_side_n+.5*error+.2*state.integral_error[side])
                offset=min(.0015,desired_force/actuator_kp+(1e-5 if slip_velocity>.003 else 0.))
                targets[side]=max(limits[side][0],actual[side]-offset) if side==0 else min(limits[side][1],actual[side]+offset)
            if saturated:state.saturated_steps+=1
        state.gripper_targets=(float(targets[0]),float(targets[1]))
        self.trace.append({'phase':phase,'retention_budget':state.budget.as_dict(),'adaptive_force_target_per_side_n':state.adaptive_force_per_side_n,'tool_normal_force_n':force,'tool_tangential_force_n':[metrics['left']['tangential_n'],metrics['right']['tangential_n']],'pad_normal_force_n':[pad_metrics['left']['normal_n'],pad_metrics['right']['normal_n']],'minimum_contact_distance_m':penetration,'relative_pose_drift_m':drift,'slip_velocity_mps':slip_velocity,'gripper_targets_m':list(state.gripper_targets),'jaw_aperture_m':float(obs.gripper_position[0]-obs.gripper_position[1]-.01),'abort_reason':state.abort_reason})
        return state.abort_reason is None
    def force_hold(self,q,state,steps,phase='force_hold'):
        for _ in range(steps):
            if not self.force_hold_step(q,state,phase):return False
        return state.bilateral_steps>=10
    def move_with_force_hold(self,q,state,phase,steps=180,frame_cb=None):
        start=np.asarray(self.robot.observe().joint_position)
        for i in range(steps):
            u=(i+1)/steps;u=u*u*(3-2*u);target=start+(np.asarray(q)-start)*u
            if not self.force_hold_step(target,state,phase):return False
            if frame_cb and i%2==0:frame_cb(phase)
        return True
    def cartesian_path_force_hold(self,start_xyz,end_xyz,state,phase,segments=12,steps_per_segment=100,frame_cb=None):
        q=np.asarray(self.robot.observe().joint_position);err=0.
        for point in np.linspace(np.asarray(start_xyz),np.asarray(end_xyz),segments+1)[1:]:
            q,err=self.solve(point,q)
            if not self.move_with_force_hold(q,state,phase,steps_per_segment,frame_cb):return q,err,False
        return q,err,True
    def cartesian_path(self,start_xyz,end_xyz,grip,phase,segments=12,steps_per_segment=100,frame_cb=None):
        q=np.asarray(self.robot.observe().joint_position);err=0.
        for point in np.linspace(np.asarray(start_xyz),np.asarray(end_xyz),segments+1)[1:]:
            q,err=self.solve(point,q);self.move(q,grip,phase,steps_per_segment,frame_cb)
        return q,err
    def bilateral_tool_contact(self,min_force_n=.05):
        sides=set()
        for i,c in enumerate(self.robot.data.contact):
            pair={int(c.geom1),int(c.geom2)}
            if self.robot._parcel not in pair:continue
            tool=pair&(self.robot._pads|self.robot._lips)
            if not tool:continue
            force=np.zeros(6);mujoco.mj_contactForce(self.m,self.robot.data,i,force)
            if abs(float(force[0]))<min_force_n:continue
            for geom in tool:
                body=self.m.body(int(self.m.geom_bodyid[geom])).name
                if body in {'left_jaw','right_jaw'}:sides.add(body)
        return sides=={'left_jaw','right_jaw'}
    def run(self,frame_cb=None):
        self.robot.reset();home=np.asarray(self.robot.observe().joint_position);open_grip=(.075,-.075)
        poses={};q=home
        for name,target in [('pregrasp',(.52,0,.945)),('insertion',GRASP_INSERTION),('lift',(.52,0,1.10)),('transport',(CARTON[0],CARTON[1],1.10)),('lower',(CARTON[0],CARTON[1],.82)),('retreat',(CARTON[0],CARTON[1],1.10))]:
            q,err=self.solve(target,q);poses[name]={'q':q,'error_m':err}
        self.move(poses['pregrasp']['q'],open_grip,'pregrasp',500,frame_cb);self.move(poses['insertion']['q'],open_grip,'insertion',400,frame_cb);held=self.close_guarded(poses['insertion']['q'],frame_cb)
        if held is None:return {'success':False,'failure_phase':'contact','contact_failure':self.last_close_failure,'poses':{k:{'error_m':v['error_m']} for k,v in poses.items()},'trace':self.trace}
        # Each 13.3 mm lift segment uses a 140 ms smoothstep.  Its analytical
        # peak acceleration is about 4.1 m/s^2; budget 5 m/s^2 with margin.
        budget=calculate_retention_budget(self.robot.object_mass_kg,5.,self.robot.effective_pad_friction(),tool_force_cap_n=125.,fragility_force_ceiling_n=125.)
        if not budget.feasible:return {'success':False,'failure_phase':'retention_budget','retention_budget':budget.as_dict(),'trace':self.trace}
        hold_q=np.asarray(self.robot.observe().joint_position);force_state=self.start_force_hold(held,budget)
        if not self.force_hold(hold_q,force_state,250,'preload'):return {'success':False,'failure_phase':'preload','retention_abort':force_state.abort_reason,'retention_budget':budget.as_dict(),'trace':self.trace}
        lift_q,lift_err,lift_ok=self.cartesian_path_force_hold(GRASP_INSERTION,(.52,0,1.10),force_state,'lift',24,70,frame_cb);poses['lift']={'q':lift_q,'error_m':lift_err}
        if not lift_ok:return {'success':False,'failure_phase':'lift_retention','retention_abort':force_state.abort_reason,'retention_budget':budget.as_dict(),'trace':self.trace}
        transport_q,transport_err,transport_ok=self.cartesian_path_force_hold((.52,0,1.10),(CARTON[0],CARTON[1],1.10),force_state,'transport',24,80,frame_cb);poses['transport']={'q':transport_q,'error_m':transport_err}
        if not transport_ok or not self.bilateral_tool_contact():
            return {'success':False,'failure_phase':'transport_retention','retention_abort':force_state.abort_reason,'retention_budget':budget.as_dict(),'poses':{k:{'error_m':v['error_m']} for k,v in poses.items()},'object_pose_writes_after_reset':False,'trace':self.trace}
        lower_q,lower_err=self.cartesian_path((CARTON[0],CARTON[1],1.10),(CARTON[0],CARTON[1],.82),held,'lower',20,50,frame_cb);poses['lower']={'q':lower_q,'error_m':lower_err};self.move(np.asarray(self.robot.observe().joint_position),open_grip,'release',160,frame_cb)
        retreat_q,retreat_err=self.solve((CARTON[0],CARTON[1],1.10),np.asarray(self.robot.observe().joint_position));poses['retreat']={'q':retreat_q,'error_m':retreat_err};self.move(retreat_q,open_grip,'retreat',500,frame_cb)
        return {'success':bool(self.robot.verify()['camera_estimate_inside_box'] and self.robot.scorer_state()['inside_box']),'failure_phase':None,'poses':{k:{'error_m':v['error_m']} for k,v in poses.items()},'camera_verification':self.robot.verify(),'scorer_verification':self.robot.scorer_state()['inside_box'],'object_pose_writes_after_reset':False,'trace':self.trace}
