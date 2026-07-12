"""Find a geometry-derived open reset transform for compatible parcel widths."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import ROOT, make_scene

def run(width_m: float) -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/reset_clearance_map_scene.xml"; make_scene(scene); scene.write_text(scene.read_text().replace('size=".06 .05 .07"', f'size=".06 {width_m / 2:.6f} .07"'))
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"); obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel"); qobj = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")]); d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]; mujoco.mj_forward(m, d); d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]] = .04; d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")]] = .04; mujoco.mj_forward(m, d)
        left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"); right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger"); finger = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in {left, right} and int(m.geom_contype[g]) > 0]
        # The derived overlay's contact pads are the small box geoms; all other
        # active finger geoms remain structural/prohibited.
        pads = [g for g in finger if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX) and float(m.geom_size[g][0]) <= .0031 and float(m.geom_size[g][1]) <= .0021]
        structural = [g for g in finger if g not in pads]
        rotation = np.asarray(d.site_xmat[site]).reshape(3, 3); origin = np.asarray(d.site_xpos[site]).copy()
        rows = []
        for x in np.linspace(-.012, .012, 25):
            for y in np.linspace(-.012, .012, 25):
                for z in np.linspace(-.012, .012, 25):
                    center = origin + rotation @ np.array([x, y, z]); d.qpos[qobj:qobj + 3] = center; mujoco.mj_forward(m, d)
                    def dist(g): return float(mujoco.mj_geomDistance(m, d, g, obj, 1.0, np.zeros(6)))
                    def points(g):
                        fromto = np.zeros(6); distance = float(mujoco.mj_geomDistance(m, d, g, obj, 1.0, fromto)); return distance, {"finger_m": fromto[:3].tolist(), "parcel_m": fromto[3:].tolist()}
                    pd = {"left": min((dist(g), g) for g in pads if int(m.geom_bodyid[g]) == left), "right": min((dist(g), g) for g in pads if int(m.geom_bodyid[g]) == right)}
                    sd = min((dist(g), g) for g in structural)
                    pad_min = min(pd["left"][0], pd["right"][0]); symmetry = abs(pd["left"][0] - pd["right"][0])
                    finger_contacts = sum(1 for ci in range(d.ncon) if obj in {int(d.contact[ci].geom1), int(d.contact[ci].geom2)} and ({int(d.contact[ci].geom1), int(d.contact[ci].geom2)} & set(finger)))
                    rows.append({"offset_grasp_frame_m": [x, y, z], "object_center_world_m": center.tolist(), "pad_clearance_m": {k: v[0] for k, v in pd.items()}, "pad_geom_ids": {k: v[1] for k, v in pd.items()}, "structural_clearance_m": sd[0], "structural_geom_id": sd[1], "closest_points_world_m": {"left_pad": points(pd["left"][1])[1], "right_pad": points(pd["right"][1])[1], "structural": points(sd[1])[1]}, "min_clearance_m": min(pad_min, sd[0]), "pad_symmetry_m": symmetry, "active_contact_count": finger_contacts, "support_contact_count": int(d.ncon) - finger_contacts})
        valid = [r for r in rows if min(r["pad_clearance_m"].values()) >= .002 and r["structural_clearance_m"] >= .002 and r["pad_symmetry_m"] <= .001 and r["active_contact_count"] == 0]
        symmetric = [r for r in rows if min(r["pad_clearance_m"].values()) >= .002 and r["structural_clearance_m"] >= .002 and r["pad_symmetry_m"] <= .002 and r["active_contact_count"] == 0]
        best = max(valid, key=lambda r: (r["min_clearance_m"], -sum(abs(v) for v in r["offset_grasp_frame_m"]))) if valid else None
        near = max(rows, key=lambda r: (r["min_clearance_m"], -sum(abs(v) for v in r["offset_grasp_frame_m"])))
        best_symmetric = max(symmetric, key=lambda r: (r["min_clearance_m"], -sum(abs(v) for v in r["offset_grasp_frame_m"]))) if symmetric else None
        return {"width_m": width_m, "pad_geom_ids": pads, "structural_geom_ids": structural, "grid_count": len(rows), "valid_count": len(valid), "valid_offset_keys": [tuple(round(v, 6) for v in r["offset_grasp_frame_m"]) for r in valid], "symmetric_within_2mm_count": len(symmetric), "best": best, "best_symmetric_within_2mm": best_symmetric, "best_near_miss": near, "best_closest_pairs": None if best is None else {"pad": {k: {"geom_id": best["pad_geom_ids"][k], "distance_m": best["pad_clearance_m"][k]} for k in ("left", "right")}, "structural": {"geom_id": best["structural_geom_id"], "distance_m": best["structural_clearance_m"]}}, "perturbation_robust": False}
    finally: scene.unlink(missing_ok=True)

if __name__ == "__main__":
    controls = [run(.05), run(.06)]
    # A shared rule is required; do not select independent hand-tuned offsets.
    shared_keys = sorted(set(controls[0]["valid_offset_keys"]) & set(controls[1]["valid_offset_keys"]))
    shared = bool(shared_keys)
    valid_sets = [set(tuple(k) for k in c["valid_offset_keys"]) for c in controls]
    robust_keys = []
    for key in shared_keys:
        neighbors = [(round(key[0] + dx, 6), round(key[1] + dy, 6), round(key[2] + dz, 6)) for dx in (-.001, 0, .001) for dy in (-.001, 0, .001) for dz in (-.001, 0, .001)]
        if all(set(neighbors) <= s for s in valid_sets): robust_keys.append(key)
    result = {"schema": "panda_reset_clearance_map_v1", "controls": controls, "shared_transform_rule_found": shared, "shared_valid_offset_keys": shared_keys, "robust_shared_offset_keys": robust_keys, "requirements": {"minimum_clearance_m": .002, "pad_symmetry_m": .001, "pose_perturbation_m": .001}}
    out = ROOT / "artifacts/panda_reset_clearance_map.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "summary": [{"width_m": x["width_m"], "valid_count": x["valid_count"], "best": x["best"]} for x in controls], "shared_transform_rule_found": shared}, indent=2))
