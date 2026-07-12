"""Control-ownership and actuator/aperture tracking audit."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import ROOT, make_scene

def audit_empty() -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/gripper_control_audit_scene.xml"; make_scene(scene)
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
        qf = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ("finger_joint1", "finger_joint2")]
        lb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"); rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        geoms = {s: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and int(m.geom_contype[g]) > 0] for s, b in (("left", lb), ("right", rb))}
        d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]; mujoco.mj_forward(m, d)
        obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel"); qobj = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")]); d.qpos[qobj:qobj + 3] = [2, 2, 2]; mujoco.mj_forward(m, d)
        rows = []
        for command in np.linspace(255, 0, 18):
            for _ in range(80):
                d.ctrl[7] = float(command); before = d.qpos[qf].copy(); mujoco.mj_step(m, d)
                separation = float(np.linalg.norm(np.asarray(d.geom_xpos[geoms["left"][0]]) - np.asarray(d.geom_xpos[geoms["right"][0]])))
                rows.append({"command": float(command), "joint_position_m": d.qpos[qf].tolist(), "fingertip_separation_m": separation, "joint_velocity_mps": d.qvel[qf].tolist(), "actuator_force_n": float(d.qfrc_actuator[7]), "joint_delta_m": float(np.linalg.norm(d.qpos[qf] - before)), "mj_step_calls_since_guard": 1})
        by_command = []
        for command in sorted({r["command"] for r in rows}, reverse=True):
            rs = [r for r in rows if r["command"] == command][-10:]
            by_command.append({"command": command, "settled_aperture_m": float(np.mean([r["fingertip_separation_m"] for r in rs])), "max_step_aperture_change_m": max((r["joint_delta_m"] for r in rs), default=0.0), "joint_positions_m": rs[-1]["joint_position_m"]})
        return {"control_writer": "audit_panda_gripper_control.py:single d.ctrl[7] assignment", "rows": rows, "settled_mapping": by_command, "actuator_ctrlrange": m.actuator_ctrlrange[7].tolist(), "finger_geoms": geoms}
    finally: scene.unlink(missing_ok=True)

def audit_centered(width: float = .05) -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/gripper_control_audit_scene.xml"; make_scene(scene); scene.write_text(scene.read_text().replace('size=".06 .05 .07"', f'size=".03 {width / 2:.6f} .035"'))
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m); site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"); obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel"); qobj = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")]); d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]; mujoco.mj_forward(m, d); d.qpos[qobj:qobj + 3] = d.site_xpos[site]; mujoco.mj_forward(m, d)
        lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"); rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger"); fg = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in {lf, rf} and int(m.geom_contype[g]) > 0}; rows = []; first_cross = None; abort = None; ctrl = 255.0
        for step in range(180):
            if step >= 20: ctrl = max(0.0, ctrl - 1.5)
            d.ctrl[7] = ctrl; mujoco.mj_step(m, d); distances = [float(mujoco.mj_geomDistance(m, d, g, obj, 1.0, np.zeros(6))) for g in fg]; min_dist = min(distances)
            if first_cross is None and min_dist < -.002: first_cross = step
            if abort is None and min_dist < -.002: abort = step
            rows.append({"step": step, "requested_command": ctrl, "actual_joint_positions_m": [float(d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]) for n in ("finger_joint1", "finger_joint2")], "fingertip_separation_m": float(np.linalg.norm(d.geom_xpos[min(fg)] - d.geom_xpos[max(fg)])), "left_right_signed_distance_m": distances, "joint_velocity_mps": [float(d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]]) for n in ("finger_joint1", "finger_joint2")], "actuator_force_n": float(d.qfrc_actuator[7]), "mj_step_calls_since_guard": 1})
            if abort is not None: break
        return {"width_m": width, "first_threshold_crossing_step": first_cross, "abort_step": abort, "guard_latency_steps": None if first_cross is None or abort is None else abort - first_cross, "rows": rows, "object_pose_writes_after_reset": False}
    finally: scene.unlink(missing_ok=True)

if __name__ == "__main__":
    result = {"schema": "panda_gripper_control_audit_v1", "single_writer_search": ["scripts/run_panda_contact_controller.py", "scripts/audit_panda_gripper_control.py"], "empty": audit_empty(), "centered_50mm": audit_centered(.05)}
    out = ROOT / "artifacts/panda_gripper_control_audit.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "settled_mapping": result["empty"]["settled_mapping"], "centered": {k: result["centered_50mm"][k] for k in ("first_threshold_crossing_step", "abort_step", "guard_latency_steps")}}, indent=2))
