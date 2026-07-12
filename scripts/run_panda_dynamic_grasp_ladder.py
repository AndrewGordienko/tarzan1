"""Contact-valid dynamic controls for the compatible Panda widths."""
from pathlib import Path
import json
import sys
import mujoco
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import ROOT, make_scene

def run_width(width_m: float) -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/franka_panda_dynamic_grasp_scene.xml"
    make_scene(scene); scene.write_text(scene.read_text().replace('size=".06 .05 .07"', f'size=".03 {width_m / 2:.6f} .035"'))
    try:
        m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"); obj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "parcel")
        qj = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]) for i in range(1, 8)]
        qobj = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")])
        left = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_finger"); right = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        fingers = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in {left, right} and int(m.geom_contype[g]) > 0}
        d.qpos[:7] = [0, -.5, 0, -2, 0, 1.5, .7]; mujoco.mj_forward(m, d)
        # Reset-only initialization for the centered positive control.
        d.qpos[qobj:qobj + 3] = d.site_xpos[site]; mujoco.mj_forward(m, d)
        initial = np.asarray(d.geom_xpos[obj]).copy(); rows = []
        def step(phase: str, target_q: np.ndarray, grip: float, n: int):
            for _ in range(n):
                d.ctrl[:7] = target_q; d.ctrl[7] = grip; mujoco.mj_step(m, d)
                normals = []; pairs = []
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    if int(c.geom1) in fingers and int(c.geom2) == obj or int(c.geom2) in fingers and int(c.geom1) == obj:
                        force = np.zeros(6); mujoco.mj_contactForce(m, d, ci, force); normals.append(abs(float(force[0]))); pairs.append([mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2))])
                rows.append({"phase": phase, "object_position_m": np.asarray(d.geom_xpos[obj]).tolist(), "object_height_m": float(d.geom_xpos[obj][2]), "object_linear_velocity_mps": float(np.linalg.norm(d.qvel[-6:-3])), "site_position_m": np.asarray(d.site_xpos[site]).tolist(), "normal_forces_n": normals, "max_normal_force_n": max(normals or [0.0]), "pairs": pairs, "penetration_m": min([float(d.contact[ci].dist) for ci in range(d.ncon) if int(d.contact[ci].geom1) in fingers and int(d.contact[ci].geom2) == obj or int(d.contact[ci].geom2) in fingers and int(d.contact[ci].geom1) == obj] or [0.0]), "finger_actuator": float(grip)})
        target = d.qpos[qj].copy(); step("centered_closure", target, .5, 120)
        close = target.copy(); step("retained_lift", close, .5, 10)
        # A small joint-space motion tests whether contact retains the parcel;
        # no parcel state is written after the reset above.
        lift_target = target.copy(); lift_target[3] += .20; step("lateral_transport", lift_target, .5, 180)
        step("release", lift_target, 255.0, 80)
        phases = {}
        for phase in ("centered_closure", "retained_lift", "lateral_transport", "release"):
            rs = [r for r in rows if r["phase"] == phase]
            phases[phase] = {"steps": len(rs), "max_force_n": max((r["max_normal_force_n"] for r in rs), default=0.0), "max_height_m": max((r["object_height_m"] for r in rs), default=0.0), "min_penetration_m": min((r["penetration_m"] for r in rs), default=0.0)}
        retained = phases["lateral_transport"]["max_height_m"] > initial[2] + .05 and phases["lateral_transport"]["max_force_n"] > .01
        return {"width_m": width_m, "initial_object_position_m": initial.tolist(), "object_pose_writes_after_reset": False, "phases": phases, "retained_lift": bool(retained), "failure_phase": None if retained else "retained_lift", "logs": rows}
    finally:
        scene.unlink(missing_ok=True)

if __name__ == "__main__":
    result = {"schema": "panda_dynamic_grasp_ladder_v1", "controls": [run_width(.05), run_width(.06)], "penetration_limit_m": -.002, "notes": "Centered reset controls; not a table pickup or packing claim."}
    out = ROOT / "artifacts/panda_dynamic_grasp_ladder.json"; out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(out), "controls": [{"width_m": x["width_m"], "retained_lift": x["retained_lift"], "failure_phase": x["failure_phase"], "phases": x["phases"]} for x in result["controls"]]}, indent=2))
