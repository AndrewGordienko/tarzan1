"""Guarded contact-aware closure using aperture-sweep priors."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import ROOT, make_scene

def run(width_m: float) -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/contact_controller_scene.xml"
    make_scene(scene); scene.write_text(scene.read_text().replace('size=".06 .05 .07"', f'size=".03 {width_m / 2:.6f} .035"'))
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"); obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel")
        qj = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]) for i in range(1, 8)]
        qobj = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")])
        left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"); right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        fingers = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in {left, right} and int(m.geom_contype[g]) > 0}
        d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]; mujoco.mj_forward(m, d)
        # Reset-only object initialization for the centered control.
        d.qpos[qobj:qobj + 3] = d.site_xpos[site]; mujoco.mj_forward(m, d)
        initial = np.asarray(d.geom_xpos[obj]).copy(); records = []; transitions = ["open"]
        bilateral_count = 0; state = "open"; ctrl = 255.0
        for step in range(260):
            if step == 50: state = "precontact"; transitions.append(state)
            if step == 70: state = "creep"; transitions.append(state)
            if state == "creep": ctrl = max(0.0, ctrl - 1.5)
            d.ctrl[:7] = d.qpos[qj]; d.ctrl[7] = ctrl; mujoco.mj_step(m, d)
            left_d = []; right_d = []; forces = []; contact_dists = []
            for g in fingers:
                dist = float(mujoco.mj_geomDistance(m, d, g, obj, 1.0, np.zeros(6)))
                (left_d if int(m.geom_bodyid[g]) == left else right_d).append(dist)
            for ci in range(d.ncon):
                c = d.contact[ci]; pair = {int(c.geom1), int(c.geom2)}
                if obj in pair and pair & fingers:
                    force = np.zeros(6); mujoco.mj_contactForce(m, d, ci, force); forces.append(abs(float(force[0]))); contact_dists.append(float(c.dist))
            ld, rd = min(left_d), min(right_d); max_force = max(forces or [0.0]); min_dist = min(contact_dists or [0.0])
            record = {"step": step, "state": state, "actuator_command": ctrl, "finger_joint_m": [float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]) for n in ("finger_joint1", "finger_joint2")], "left_geom_distance_m": ld, "right_geom_distance_m": rd, "contact_dist_m": min_dist, "normal_force_n": max_force, "object_position_m": np.asarray(d.geom_xpos[obj]).tolist(), "object_height_m": float(d.geom_xpos[obj][2])}
            records.append(record)
            if state == "creep":
                bilateral = ld <= 0 and rd <= 0 and min_dist >= -.002 and max_force <= 70.0
                bilateral_count = bilateral_count + 1 if bilateral else 0
                if max_force > 70.0 or min_dist < -.002 or np.linalg.norm(np.asarray(d.geom_xpos[obj]) - initial) > .08:
                    state = "abort_reopen"; transitions.append(state); ctrl = 255.0
                elif bilateral_count >= 5:
                    state = "bilateral_contact"; transitions.append(state); hold_ctrl = ctrl; break
        success = state == "bilateral_contact"
        return {"width_m": width_m, "state": state, "success": success, "transitions": transitions, "actuator_open_command": 255.0, "max_observed_force_n": max((r["normal_force_n"] for r in records), default=0.0), "min_observed_contact_dist_m": min((r["contact_dist_m"] for r in records), default=0.0), "initial_object_position_m": initial.tolist(), "object_pose_writes_after_reset": False, "records": records}
    finally:
        scene.unlink(missing_ok=True)

if __name__ == "__main__":
    result = {"schema": "panda_contact_controller_v1", "controls": [run(.05), run(.06)], "force_limit_n": 70.0, "penetration_limit_m": -.002, "required_persistent_steps": 5}
    out = ROOT / "artifacts/panda_contact_controller.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "controls": [{"width_m": x["width_m"], "state": x["state"], "success": x["success"], "transitions": x["transitions"], "max_force_n": x["max_observed_force_n"], "min_contact_dist_m": x["min_observed_contact_dist_m"]} for x in result["controls"]]}, indent=2))
