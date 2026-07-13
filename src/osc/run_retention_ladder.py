"""Matched retention attribution for the light-sortable gripper lane."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from osc.packcell.retention import calculate_retention_budget
from osc.packcell.ur10e_adapter import GRASP_INSERTION, UR10eAdapter
from osc.packcell.ur10e_scripted import UR10eScriptedController
from osc.robot_api import ActuatorCommand

CHECKPOINTS_S=(.5,1.,2.,5.,10.)


def setup_case(mass_kg,gravity_enabled,support_lip):
    robot=UR10eAdapter(support_lip=support_lip,telemetry=True,object_mass_kg=mass_kg,gravity_enabled=gravity_enabled)
    controller=UR10eScriptedController(robot);robot.reset();q=np.asarray(robot.observe().joint_position);open_grip=(.075,-.075)
    for phase,target,steps in [('pregrasp',(.52,0,.945),500),('insertion',GRASP_INSERTION,400)]:
        q,_=controller.solve(target,q);controller.move(q,open_grip,phase,steps)
    held=controller.close_guarded(q)
    return robot,controller,q,held


def observed_row(robot,state,phase):
    tool=robot.tool_contact_metrics(pads_only=False);pads=robot.tool_contact_metrics(pads_only=True);relative=robot.parcel_relative_to_grasp_site();drift=float(np.linalg.norm(relative-state.initial_relative_pose))
    return {'phase':phase,'time_s':float(robot.data.time),'left_normal_n':tool['left']['normal_n'],'right_normal_n':tool['right']['normal_n'],'left_tangential_n':tool['left']['tangential_n'],'right_tangential_n':tool['right']['tangential_n'],'left_pad_normal_n':pads['left']['normal_n'],'right_pad_normal_n':pads['right']['normal_n'],'bilateral':tool['bilateral'],'minimum_distance_m':tool['minimum_distance_m'],'relative_pose_drift_m':drift,'jaw_aperture_m':float(robot.data.qpos[6]-robot.data.qpos[7]-.01),'adaptive_force_target_per_side_n':state.adaptive_force_per_side_n,'abort_reason':state.abort_reason}


def run_case(mass_kg,mode,gravity_enabled=True,support_lip=True):
    robot,controller,q,held=setup_case(mass_kg,gravity_enabled,support_lip);mu=robot.effective_pad_friction();budget=calculate_retention_budget(mass_kg,5.,mu,gravity_mps2=9.81 if gravity_enabled else 0.)
    result={'mass_kg':mass_kg,'mode':mode,'gravity_enabled':gravity_enabled,'support_lip':support_lip,'configured_friction':{'pad':2.0,'parcel':.8,'effective':mu},'budget':budget.as_dict(),'custom_fingertip_dependency_label':'UR10e-class arm + 2F-140-equivalent gripper with custom support fingertips' if support_lip else 'flat-pad ablation','checkpoints':{str(x):False for x in CHECKPOINTS_S},'curve':[]}
    def finish(**updates):
        result.update(updates);result['prediction_observation']='predicted_success_observed_failure' if budget.feasible and result.get('failure_phase') else ('predicted_success_observed_success' if budget.feasible else 'predicted_failure')
        result['_trace']=controller.trace.copy();return result
    if held is None:return finish(failure_phase='bilateral_contact_setup',failure_reason=controller.last_close_failure or 'opposing_contact_not_established')
    state=controller.start_force_hold(held,budget)
    if not controller.force_hold(q,state,250,'preload'):return finish(failure_phase='preload',failure_reason=state.abort_reason)
    if mode=='force_hold':
        _,_,ok=controller.cartesian_path_force_hold(GRASP_INSERTION,(.52,0,1.10),state,'lift',24,70)
    else:
        frozen=state.gripper_targets;controller.cartesian_path(GRASP_INSERTION,(.52,0,1.10),frozen,'lift',24,70);tool=robot.tool_contact_metrics();drift=float(np.linalg.norm(robot.parcel_relative_to_grasp_site()-state.initial_relative_pose));ok=controller.bilateral_tool_contact() and (tool['minimum_distance_m'] is None or tool['minimum_distance_m']>=-.002) and drift<.02
    if not ok:return finish(failure_phase='lift',failure_reason=state.abort_reason or ('penetration_limit' if robot.tool_contact_metrics()['minimum_distance_m'] is not None and robot.tool_contact_metrics()['minimum_distance_m']<-.002 else 'contact_not_retained'))
    q=np.asarray(robot.observe().joint_position);start_time=float(robot.data.time);unilateral=0;curve=[]
    for step in range(int(10./robot.model.opt.timestep)):
        if mode=='force_hold':ok=controller.force_hold_step(q,state,'retention_hold')
        else:
            robot.step(ActuatorCommand(tuple(q),state.gripper_targets,'retention_hold'));tool=robot.tool_contact_metrics(pads_only=False);relative=robot.parcel_relative_to_grasp_site();drift=float(np.linalg.norm(relative-state.initial_relative_pose));penetration=tool['minimum_distance_m'];unilateral=0 if tool['bilateral'] else unilateral+1;state.abort_reason='penetration_limit' if penetration is not None and penetration<-.002 else ('persistent_slip' if drift>.02 else ('bilateral_contact_lost' if unilateral>100 else None));ok=state.abort_reason is None;controller.trace.append(observed_row(robot,state,'retention_hold'))
        if step%50==0 or not ok:curve.append(observed_row(robot,state,'retention_hold'))
        elapsed=float(robot.data.time-start_time)
        for checkpoint in CHECKPOINTS_S:
            if elapsed>=checkpoint and ok:result['checkpoints'][str(checkpoint)]=True
        if not ok:break
    result['curve']=curve;result['achieved_hold_s']=float(robot.data.time-start_time);result['failure_phase']=None if result['checkpoints']['10.0'] else 'retention_hold';result['failure_reason']=None if result['failure_phase'] is None else state.abort_reason or 'duration_not_reached';result['max_force_n']=max([r['left_normal_n']+r['right_normal_n'] for r in curve] or [0]);result['minimum_distance_m']=min([r['minimum_distance_m'] for r in curve if r['minimum_distance_m'] is not None] or [0]);result['final_drift_m']=curve[-1]['relative_pose_drift_m'] if curve else None
    return finish()


def main():
    cases=[]
    for mass in (.1,.5,1.,2.):
        for support_lip in (True,False):
            cases.extend((run_case(mass,'position_hold',support_lip=support_lip),run_case(mass,'force_hold',support_lip=support_lip)))
    cases.extend((run_case(.1,'position_hold',gravity_enabled=False,support_lip=False),run_case(.1,'force_hold',gravity_enabled=False,support_lip=False)))
    result={'schema':'light_sortable_retention_ladder_v1','claim_label':'Historical lifted lane: UR10e-class arm + 2F-140-equivalent gripper with custom support fingertips. Corrected bounded-contact lane has not passed.','transport_trajectory_changed':False,'friction_tuned':False,'checkpoint_durations_s':list(CHECKPOINTS_S),'cases':cases,'summary':{'cases':len(cases),'ten_second_passes':sum(c['checkpoints']['10.0'] for c in cases),'force_hold_ten_second_passes':sum(c['mode']=='force_hold' and c['checkpoints']['10.0'] for c in cases),'promotion_gate_passed':False}}
    telemetry=Path('artifacts/light_sortable_retention_curves.jsonl.gz')
    with telemetry.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            for case_index,case in enumerate(cases):
                for row in case.pop('_trace',[]):compressed.write((json.dumps({'case_index':case_index,**row},separators=(',',':'))+'\n').encode())
                case.pop('curve',None)
    result['curve_artifact']=str(telemetry);out=Path('artifacts/light_sortable_retention_ladder.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'artifact':str(out),'curve_artifact':str(telemetry),'summary':result['summary'],'cases':[{'mass_kg':c['mass_kg'],'mode':c['mode'],'gravity_enabled':c['gravity_enabled'],'support_lip':c['support_lip'],'failure_phase':c.get('failure_phase'),'failure_reason':c.get('failure_reason'),'checkpoints':c['checkpoints']} for c in cases]},indent=2))


if __name__=='__main__':main()
