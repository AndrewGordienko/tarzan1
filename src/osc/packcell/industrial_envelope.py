from __future__ import annotations
import json, numpy as np, mujoco
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; MODEL=ROOT/'assets/industrial/derived/franka_panda_tarzan/panda_grasp_overlay.xml'

def evaluate(layouts=50):
    m=mujoco.MjModel.from_xml_path(str(MODEL)); d=mujoco.MjData(m); sid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_site')
    lanes={k:{'layouts':layouts,'valid_endpoints':0,'valid_paths':0,'failures':{'reach':0,'joint_limit':0,'singularity':0,'self_collision':0,'table_collision':0,'box_collision':0},'status':'endpoint_search_not_started'} for k in ('free_space','pick_only','placement_only','full_workcell')}
    jacobian=[]
    for label in ('home','pregrasp','insertion','placement'):
        mujoco.mj_resetData(m,d); mujoco.mj_forward(m,d); J=np.zeros((3,m.nv)); mujoco.mj_jacSite(m,d,J,None,sid); sv=np.linalg.svd(J,compute_uv=False); jacobian.append({'pose':label,'rank':int(np.linalg.matrix_rank(J)),'singular_values':sv.tolist(),'finite':bool(np.isfinite(J).all()),'min_singular_value':float(sv.min()),'condition_number':float(sv.max()/max(sv.min(),1e-12))})
    return {'asset':str(MODEL.relative_to(ROOT)),'development_layouts':layouts,'jacobian_gates':jacobian,'lanes':lanes,'confirmation_layouts_sealed':100,'status':'geometry_variant_evaluator_pending'}
