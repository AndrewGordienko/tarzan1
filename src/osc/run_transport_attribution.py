from pathlib import Path
import gzip, json, math
import numpy as np
from collections import Counter
from osc.packcell.ur10e_adapter import CARTON, UR10eAdapter
from osc.packcell.ur10e_scripted import UR10eScriptedController

def prepare(robot,controller):
    robot.reset();q=np.asarray(robot.observe().joint_position);open_grip=(.075,-.075)
    for phase,target,steps in [('pregrasp',(.52,0,.93),500),('insertion',(.52,0,.78),400)]:q,_=controller.solve(target,q);controller.move(q,open_grip,phase,steps)
    held=controller.close_guarded(q)
    if held is None:return None
    controller.move(np.asarray(robot.observe().joint_position),held,'grasp_dwell',200)
    controller.cartesian_path((.52,0,.78),(.52,0,1.10),held,'lift',24,70)
    return held

def summarize(robot,phase):
    rows=[r for r in robot.telemetry if r['controller_state']==phase];metric_rows=rows[5:] if len(rows)>5 else rows;initial_z=rows[0]['parcel_position_m'][2] if rows else 0
    contacts=[c for r in rows for c in r['contacts'] if c['kind']=='tool_parcel']
    final_contact=bool(rows and len(rows)>=10 and all(any(c['kind']=='tool_parcel' for c in r['contacts']) for r in rows[-10:]))
    relative_drift=float(np.linalg.norm(np.asarray(rows[-1]['parcel_relative_to_gripper_m'])-np.asarray(rows[0]['parcel_relative_to_gripper_m']))) if rows else 0.
    retained=bool(rows and rows[-1]['parcel_position_m'][2]>.83 and final_contact and relative_drift<.02)
    classes=Counter(c['collision_class'] for r in rows for c in r['contacts'] if c.get('prohibited'))
    return {'steps':len(rows),'retained':retained,'final_contact_retained':final_contact,'relative_pose_drift_m':relative_drift,'initial_z_m':initial_z,'final_z_m':rows[-1]['parcel_position_m'][2] if rows else None,'max_normal_force_n':max([c['normal_n'] for c in contacts] or [0]),'max_tangential_force_n':max([c['tangential_n'] for c in contacts] or [0]),'max_friction_cone_utilization':max([c['friction_cone_utilization'] for c in contacts if c['normal_n']>.01] or [0]),'max_linear_slip_mps':max([float(np.linalg.norm(r['linear_slip_mps'])) for r in metric_rows] or [0]),'jaw_aperture_range_m':[min([r['jaw_aperture_m'] for r in rows] or [0]),max([r['jaw_aperture_m'] for r in rows] or [0])],'max_jaw_actuator_force_n':max([max(abs(x) for x in r['jaw_actuator_force_n']) for r in rows] or [0]),'max_tcp_acceleration_mps2':max([float(np.linalg.norm(r['tcp_acceleration_mps2'])) for r in metric_rows] or [0]),'max_tcp_jerk_mps3':max([float(np.linalg.norm(r['tcp_jerk_mps3'])) for r in metric_rows] or [0]),'max_angular_velocity_rps':max([float(np.linalg.norm(r['angular_velocity_rps'])) for r in metric_rows] or [0]),'collision_impulse_ns':sum(r['collision_impulse_ns'] for r in rows),'prohibited_contact_steps':sum(any(c.get('prohibited') for c in r['contacts']) for r in rows),'prohibited_contact_counts':dict(classes),'contact_steps':sum(any(c['kind']=='tool_parcel' for c in r['contacts']) for r in rows)}

def first_divergence(lane_result):
    rows=lane_result['telemetry']
    if not rows:return {}
    baseline=rows[0]['jaw_aperture_m']
    def first(predicate):
        return next(({'step':r['step'],'time_s':r['time_s']} for r in rows if predicate(r)),None)
    def first_sustained(predicate,count=10):
        for i in range(len(rows)-count+1):
            if all(predicate(r) for r in rows[i:i+count]):
                return {'step':rows[i]['step'],'time_s':rows[i]['time_s'],'sustained_steps':count}
        return None
    return {
        'jaw_aperture_changed_over_10mm':first(lambda r:abs(r['jaw_aperture_m']-baseline)>.010),
        'jaw_force_saturated':first(lambda r:max(abs(x) for x in r['jaw_actuator_force_n'])>=62.49),
        'tool_parcel_contact_lost':first_sustained(lambda r:not any(c['kind']=='tool_parcel' for c in r['contacts'])),
        'prohibited_contact':first(lambda r:any(c.get('prohibited') for c in r['contacts'])),
        'parcel_below_retention_height':first(lambda r:r['parcel_position_m'][2]<=.83),
    }

def lane(name,kind,steps_per_segment=100,support_lip=True):
    robot=UR10eAdapter(support_lip=support_lip,telemetry=True);controller=UR10eScriptedController(robot);held=prepare(robot,controller)
    if held is None:return {'lane':name,'setup_success':False,'failure':'grasp','summary':{},'telemetry':robot.telemetry}
    start=len(robot.telemetry)
    if kind=='hold':controller.move(np.asarray(robot.observe().joint_position),held,'test',steps_per_segment)
    elif kind=='translate':controller.cartesian_path((.52,0,1.10),(.52,.10,1.10),held,'test',10,steps_per_segment)
    elif kind=='rotate':
        q=np.asarray(robot.observe().joint_position);q[-1]+=math.radians(20);controller.move(q,held,'test',600)
    elif kind=='planned':controller.cartesian_path((.52,0,1.10),(CARTON[0],CARTON[1],1.10),held,'test',24,steps_per_segment)
    result={'lane':name,'setup_success':True,'support_lip':support_lip,'profile':{'kind':kind,'steps_per_segment':steps_per_segment},'summary':summarize(robot,'test'),'telemetry':robot.telemetry[start:]};result['failure']=None if result['summary']['retained'] else 'transport_retention';return result

def main():
    lanes=[lane('stationary_hold','hold',600),lane('stationary_hold_matched_duration','hold',4800),lane('translation_very_low','translate',200),lane('translation_low','translate',100),lane('translation_medium','translate',60),lane('translation_nominal','translate',35),lane('wrist_rotation','rotate'),lane('complete_planned_transfer_very_low','planned',200),lane('complete_planned_transfer','planned',80),lane('translation_very_low_no_support_lip','translate',200,False)]
    full=lanes[8];no_lip=lanes[9]
    result={'schema':'light_sortable_transport_attribution_v1','identical_grasp_setup':True,'friction_changed_for_ladder':False,'lanes':lanes,'diagnosis':{'short_hold_pass':lanes[0]['summary'].get('retained'),'matched_duration_hold_pass':lanes[1]['summary'].get('retained'),'slow_translation_pass':lanes[2]['summary'].get('retained'),'nominal_translation_pass':lanes[5]['summary'].get('retained'),'wrist_rotation_pass':lanes[6]['summary'].get('retained'),'slow_planned_transfer_pass':lanes[7]['summary'].get('retained'),'planned_transfer_pass':full['summary'].get('retained'),'no_lip_grasp_setup_pass':no_lip.get('setup_success'),'no_lip_slow_pass':no_lip['summary'].get('retained'),'support_lip_ablation_result':'setup_failed_without_lip' if not no_lip.get('setup_success') else ('retained' if no_lip['summary'].get('retained') else 'transport_failed'),'full_transfer_first_divergence':first_divergence(full),'acceleration_only_explanation_supported':bool(lanes[2]['summary'].get('retained') and not lanes[5]['summary'].get('retained')),'dominant_failure':'static_retention_creep' if not lanes[1]['summary'].get('retained') else 'unresolved_dynamic_transport','allowable_acceleration_fit':None,'allowable_acceleration_fit_reason':'matched_duration_stationary_hold_fails; static retention must pass before fitting a dynamic acceleration bound','one_arm_acceptance_gate_passed':False}}
    telemetry_path=Path('artifacts/light_sortable_transport_telemetry.jsonl.gz')
    with telemetry_path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            for lane_result in lanes:
                for row in lane_result['telemetry']:
                    compressed.write((json.dumps({'lane':lane_result['lane'],**row},separators=(',',':'))+'\n').encode())
    for lane_result in lanes:
        lane_result['telemetry_rows']=len(lane_result.pop('telemetry'))
        lane_result['telemetry_artifact']=str(telemetry_path)
    result['telemetry_artifact']=str(telemetry_path)
    out=Path('artifacts/light_sortable_transport_attribution.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'artifact':str(out),'diagnosis':result['diagnosis'],'summaries':{x['lane']:x['summary'] for x in lanes}},indent=2))
if __name__=='__main__':main()
