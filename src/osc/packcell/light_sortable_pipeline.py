"""Bridge Tarzan planning decisions into hardware-independent skills."""
from __future__ import annotations
from osc.packcell.grasp_feasibility import EndEffectorContract, select_grasp_axis
from osc.packcell.grasp_pose_generator import generate_grasp_pose
from osc.packing.domain import Container, PackItem, PackingState
from osc.packing.planner import PackingPlanner
from osc.packcell.retention import calculate_retention_budget

CONTRACT=EndEffectorContract('simplified_2f140_equivalent',0.,.140,.003,(.05,.01,.08),'+Y tool frame',(10.,125.),2.5,1.2,(0,1,2),('glass','ceramic','unknown'))

def compile_item_execution(dimensions_m, mass_kg, *, planned_acceleration_mps2=1.0, friction_coefficient=.8, fragility_force_ceiling_n=125.):
    dims=tuple(float(x) for x in dimensions_m)
    if mass_kg>2.:return {'status':'abstain','reason':'tool_change_required:overweight'}
    axis=select_grasp_axis(dims,CONTRACT)
    if axis is None:return {'status':'abstain','reason':'tool_change_required:aperture'}
    margin=(CONTRACT.max_usable_aperture_m-dims[axis])/2-CONTRACT.insertion_clearance_m
    pose=generate_grasp_pose([{'offset_grasp_frame_m':[0,0,0],'structural_clearance_m':margin,'left_pad_clearance_m':margin,'right_pad_clearance_m':margin,'pad_overlap_m':min(dims),'robust_clearance_m':margin-.001}])
    if pose['status']!='selected':return {'status':'abstain','reason':'no_robust_grasp_pose'}
    retention=calculate_retention_budget(mass_kg,planned_acceleration_mps2,friction_coefficient,tool_force_cap_n=CONTRACT.grip_force_range_n[1],fragility_force_ceiling_n=fragility_force_ceiling_n)
    if not retention.feasible:return {'status':'abstain','reason':f'retention_incompatible:{retention.rejection_reason}','retention_budget':retention.as_dict()}
    item=PackItem('parcel',dims,mass_kg);state=PackingState(Container('carton',(.32,.24,.20),2.5),{'parcel':item});plan=PackingPlanner().plan(state)
    if not plan.feasible:return {'status':'abstain','reason':'no_packing_plan'}
    placement=plan.actions[0].placement
    return {'status':'execute','selected_grasp_axis':axis,'grasp_pose':pose['pose'],'retention_budget':retention.as_dict(),'placement':{'position':list(placement.position),'size':list(placement.size),'orientation':list(placement.orientation)},'skills':['approach','grasp','preload','force_hold','lift','transport','place','release','verify']}
