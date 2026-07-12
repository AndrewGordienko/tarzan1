"""Quasi-static centered-contact controls for compatible parcel widths."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import ROOT, make_scene

def sweep(width_m: float) -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/width_sweep_scene.xml"
    make_scene(scene)
    scene.write_text(scene.read_text().replace('size=".06 .05 .07"', f'size=".06 {width_m / 2:.6f} .07"'))
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel")
        qobj = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")])
        qf = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ("finger_joint1", "finger_joint2")]
        bodies = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s) for s in ("left_finger", "right_finger")}
        d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]; mujoco.mj_forward(m, d)
        d.qpos[qobj:qobj + 3] = d.site_xpos[site]; mujoco.mj_forward(m, d)
        rows = []
        for aperture in np.linspace(.04, 0, 41):
            d.qpos[qf] = aperture; d.qvel[:] = 0; mujoco.mj_forward(m, d)
            distances = {}
            for side, body in bodies.items():
                geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body and int(m.geom_contype[g]) > 0]
                distances[side] = min(float(mujoco.mj_geomDistance(m, d, g, obj, 1.0, np.zeros(6))) for g in geoms)
            rows.append({"aperture_joint_m": float(aperture), "left_distance_m": distances["left_finger"], "right_distance_m": distances["right_finger"], "symmetry_m": abs(distances["left_finger"] - distances["right_finger"])})
        valid = [r for r in rows if -0.002 <= r["left_distance_m"] <= 0 and -0.002 <= r["right_distance_m"] <= 0 and r["symmetry_m"] <= .001]
        adjacent = sum(1 for a, b in zip(rows, rows[1:]) if a in valid and b in valid)
        return {"grasp_width_m": width_m, "samples": rows, "valid_samples": len(valid), "adjacent_valid_intervals": adjacent, "promotion_pass": bool(adjacent)}
    finally:
        scene.unlink(missing_ok=True)

if __name__ == "__main__":
    result = {"schema": "panda_width_sweep_v1", "controls": [sweep(.05), sweep(.06)], "penetration_limit_m": -.002, "symmetry_limit_m": .001}
    out = ROOT / "artifacts/panda_compatible_width_sweeps.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "summary": [(x["grasp_width_m"], x["valid_samples"], x["adjacent_valid_intervals"], x["promotion_pass"]) for x in result["controls"]]}, indent=2))
