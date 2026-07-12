"""Final bounded stock-Panda grasp search and active-constraint attribution."""
from __future__ import annotations
from pathlib import Path
import itertools, json, math, sys
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import ROOT, make_scene

MIN_CLEARANCE = .002
UNCERTAINTY = .001

def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2); cp, sp = math.cos(pitch / 2), math.sin(pitch / 2); cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])

def main() -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/final_grasp_decision_scene.xml"; make_scene(scene)
    scene.write_text(scene.read_text().replace('size=".06 .05 .07"', 'size=".025 .025 .025"'))
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"); obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel"); joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free"); qobj = int(m.jnt_qposadr[joint]); d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]
        for name in ("finger_joint1", "finger_joint2"): d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = .04
        mujoco.mj_forward(m, d); origin = np.asarray(d.site_xpos[site]).copy(); R = np.asarray(d.site_xmat[site]).reshape(3, 3)
        left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"); right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        finger = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in {left, right} and int(m.geom_contype[g]) > 0]
        pads = [g for g in finger if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX) and float(m.geom_size[g][0]) <= .0031 and float(m.geom_size[g][1]) <= .0021]
        structural = [g for g in finger if g not in pads]
        def geom_name(g): return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"unnamed_geom_{g}"
        def evaluate(offset, rpy, perturb=(0,0,0)):
            d.qpos[qobj:qobj+3] = origin + R @ (np.asarray(offset) + np.asarray(perturb)); d.qpos[qobj+3:qobj+7] = quat_from_euler(*rpy); mujoco.mj_forward(m, d)
            def rec(g):
                ft=np.zeros(6); dist=float(mujoco.mj_geomDistance(m,d,g,obj,1.,ft)); return {"distance_m":dist,"geom_id":g,"geom_name":geom_name(g),"body":mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])),"geom_type":int(m.geom_type[g]),"fromto_world_m":{"finger":ft[:3].tolist(),"parcel":ft[3:].tolist()}}
            lp=min((rec(g) for g in pads if int(m.geom_bodyid[g])==left),key=lambda x:x["distance_m"]); rp=min((rec(g) for g in pads if int(m.geom_bodyid[g])==right),key=lambda x:x["distance_m"]); st=min((rec(g) for g in structural),key=lambda x:x["distance_m"])
            contacts=sum(1 for i in range(d.ncon) if obj in {int(d.contact[i].geom1),int(d.contact[i].geom2)} and ({int(d.contact[i].geom1),int(d.contact[i].geom2)}&set(finger)))
            values={"left_pad":lp,"right_pad":rp,"structural_finger":st}; binding=min(values,key=lambda k:values[k]["distance_m"])
            return {"minimum_clearance_m":values[binding]["distance_m"],"binding_class":binding,"binding":values[binding],"pad_clearance_m":{"left":lp["distance_m"],"right":rp["distance_m"]},"pad_symmetry_m":abs(lp["distance_m"]-rp["distance_m"]),"active_contacts":contacts}
        coarse=[]
        offsets=itertools.product(np.linspace(-.015,.015,7),np.linspace(-.010,.010,5),np.linspace(-.015,.015,7))
        orientations=[tuple(math.radians(v) for v in rpy) for rpy in itertools.product((-5,0,5),(-5,0,5),(0,90))]
        for off in offsets:
            for rpy in orientations:
                e=evaluate(off,rpy); coarse.append({"offset_grasp_frame_m":list(off),"rpy_rad":list(rpy),**e})
        seeds=sorted(coarse,key=lambda c:(c["minimum_clearance_m"],-c["pad_symmetry_m"]),reverse=True)[:12]
        refined=[]
        for seed in seeds:
            for delta in itertools.product((-0.002,-0.001,0,.001,.002),repeat=3):
                off=tuple(seed["offset_grasp_frame_m"][i]+delta[i] for i in range(3)); rpy=tuple(seed["rpy_rad"]); e=evaluate(off,rpy); refined.append({"offset_grasp_frame_m":list(off),"rpy_rad":list(rpy),**e})
        nominal=[c for c in refined if c["minimum_clearance_m"]>=MIN_CLEARANCE and c["pad_symmetry_m"]<=.001 and c["active_contacts"]==0]
        robust=[]; attributed=[]
        for c in sorted(nominal,key=lambda x:x["minimum_clearance_m"],reverse=True)[:100]:
            pert=[]
            for p in itertools.product((-UNCERTAINTY,0,UNCERTAINTY),repeat=3): pert.append({"direction_m":list(p),**evaluate(c["offset_grasp_frame_m"],c["rpy_rad"],p)})
            worst=min(pert,key=lambda x:x["minimum_clearance_m"]); c={**c,"worst_case":worst,"robust_clearance_m":worst["minimum_clearance_m"],"clearance_deficit_m":max(0.,MIN_CLEARANCE-worst["minimum_clearance_m"]),"pad_only_robust_clearance_m":min(min(x["pad_clearance_m"].values()) for x in pert)}; attributed.append(c)
            # Nominal closing-axis symmetry is gated above.  Under a declared
            # lateral pose perturbation the left/right clearances are expected
            # to differ; robustness requires positive clearance, not symmetry.
            if c["robust_clearance_m"]>=MIN_CLEARANCE and all(x["active_contacts"]==0 for x in pert): robust.append(c)
        best=max(attributed,key=lambda x:x["robust_clearance_m"]) if attributed else None; selected=max(robust,key=lambda x:x["robust_clearance_m"]) if robust else None
        collision_fidelity={"structural_geoms":[{"geom_id":g,"name":geom_name(g),"type":int(m.geom_type[g]),"mesh_collision":int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_MESH),"size":m.geom_size[g].tolist()} for g in structural],"conclusion":"mesh_collision_is_convexified_by_MuJoCo; physical-fidelity correction requires documented jaw dimensions before any primitive overlay","derived_geometry_changed":False}
        return {"schema":"panda_final_grasp_decision_v1","object_dimensions_m":[.05,.05,.05],"candidate_counts":{"coarse":len(coarse),"refined":len(refined),"nominal":len(nominal),"robust":len(robust)},"search":{"translations_m":{"x":[-.015,.015],"y":[-.010,.010],"z":[-.015,.015]},"rpy_deg":{"roll":[-5,0,5],"pitch":[-5,0,5],"yaw":[0,90]},"axis_flips":True,"uncertainty_m":UNCERTAINTY,"minimum_clearance_m":MIN_CLEARANCE},"active_constraint_attribution":best,"selected_robust_pose":selected,"pad_only_counterfactual_pass":bool(best and best["pad_only_robust_clearance_m"]>=MIN_CLEARANCE),"full_structure_pass":selected is not None,"collision_model_fidelity":collision_fidelity,"decision":"resume_guarded_closure" if selected else "close_stock_panda_physical_lane"}
    finally: scene.unlink(missing_ok=True)

if __name__=="__main__":
    result=main(); out=ROOT/"artifacts/panda_final_grasp_decision.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"artifact":str(out),"candidate_counts":result["candidate_counts"],"decision":result["decision"],"active_constraint":result["active_constraint_attribution"],"pad_only_counterfactual_pass":result["pad_only_counterfactual_pass"]},indent=2))
