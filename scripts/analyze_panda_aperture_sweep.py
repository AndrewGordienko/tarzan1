"""Deterministic, per-finger analysis of the Panda aperture sweep.

This is a geometry diagnostic only.  It directly sets finger joint positions on
a copied diagnostic model; it never changes the production controller path or
the upstream vendored asset.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_panda_motion_demo import make_scene  # noqa: E402


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or f"{str(kind)}:{index}"


def _crossing(apertures: np.ndarray, distances: np.ndarray) -> float | None:
    # Apertures are sampled from open to closed.  Interpolate the first sign
    # change so the reported value is not confused with a sample endpoint.
    for i in range(1, len(distances)):
        a0, a1 = float(apertures[i - 1]), float(apertures[i])
        d0, d1 = float(distances[i - 1]), float(distances[i])
        if d0 == 0:
            return a0
        if d0 * d1 <= 0 and d0 != d1:
            return a0 + (0.0 - d0) * (a1 - a0) / (d1 - d0)
    return None


def _extrapolated_crossing(apertures: np.ndarray, distances: np.ndarray) -> float | None:
    """Estimate a crossing outside the sampled range, explicitly labelled."""
    if np.all(distances < 0) or np.all(distances > 0):
        x = np.asarray(apertures[:2], dtype=float)
        y = np.asarray(distances[:2], dtype=float)
        if abs(y[1] - y[0]) > 1e-12:
            return float(x[0] + (0.0 - y[0]) * (x[1] - x[0]) / (y[1] - y[0]))
    return None


def _geom_record(model: mujoco.MjModel, data: mujoco.MjData, gid: int, obj: int):
    fromto = np.zeros(6, dtype=float)
    distance = float(mujoco.mj_geomDistance(model, data, gid, obj, 1.0, fromto))
    rot = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    # Record every local axis.  For a box/capsule, the axis most opposed to the
    # parcel-to-pad vector is the useful candidate contact normal; retaining all
    # axes makes the diagnostic auditable rather than relying on a convention.
    p0, p1 = fromto[:3], fromto[3:]
    direction = p1 - p0
    if np.linalg.norm(direction) > 1e-9:
        direction = direction / np.linalg.norm(direction)
        axis_index = int(np.argmax(np.abs(rot.T @ direction)))
    else:
        axis_index = 0
    return {
        "geom_id": int(gid),
        "geom_name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
        "body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]),
        "type": int(model.geom_type[gid]),
        "size_m": np.asarray(model.geom_size[gid], dtype=float).tolist(),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "friction": np.asarray(model.geom_friction[gid], dtype=float).tolist(),
        "distance_m": distance,
        "closest_points_world_m": {"finger": p0.tolist(), "parcel": p1.tolist()},
        "world_position_m": np.asarray(data.geom_xpos[gid], dtype=float).tolist(),
        "world_axes": rot.tolist(),
        "candidate_contact_normal_axis": axis_index,
        "candidate_contact_normal_world": rot[:, axis_index].tolist(),
    }


def run() -> dict:
    scene = ROOT / "assets/industrial/derived/franka_panda_tarzan/panda_aperture_analysis_scene.xml"
    make_scene(scene)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        body_id = {
            side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, side)
            for side in ("left_finger", "right_finger")
        }
        parcel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "parcel")
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        finger_joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("finger_joint1", "finger_joint2")
        ]
        finger_qadr = [int(model.jnt_qposadr[j]) for j in finger_joints]
        side_geoms = {
            side: [
                gid for gid in range(model.ngeom)
                if int(model.geom_bodyid[gid]) == bid and int(model.geom_contype[gid]) > 0
            ]
            for side, bid in body_id.items()
        }

        # This is deliberately the same centered diagnostic pose used by the
        # existing sweep: object initialization is allowed only at reset.
        data.qpos[:7] = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
        mujoco.mj_forward(model, data)
        parcel_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "parcel_free")
        parcel_qadr = int(model.jnt_qposadr[parcel_joint])
        data.qpos[parcel_qadr:parcel_qadr + 3] = data.site_xpos[site]
        mujoco.mj_forward(model, data)
        initial_parcel = np.asarray(data.geom_xpos[parcel], dtype=float).copy()

        # Empty-gripper control: move the diagnostic parcel away, close the
        # fingers quasi-statically, and count only finger↔finger contacts.
        data.qpos[parcel_qadr:parcel_qadr + 3] = [2.0, 2.0, 2.0]
        data.qpos[finger_qadr] = 0.0
        mujoco.mj_forward(model, data)
        empty_finger_finger = 0
        for ci in range(data.ncon):
            c = data.contact[ci]
            ba = int(model.geom_bodyid[c.geom1]); bb = int(model.geom_bodyid[c.geom2])
            if {ba, bb} == {body_id["left_finger"], body_id["right_finger"]}:
                empty_finger_finger += 1
        # Restore the centered reset state before sampling.
        data.qpos[parcel_qadr:parcel_qadr + 3] = initial_parcel
        mujoco.mj_forward(model, data)
        apertures = np.linspace(0.04, 0.0, 41)
        rows = []
        for aperture in apertures:
            data.qpos[finger_qadr] = float(aperture)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            side = {
                name: sorted((_geom_record(model, data, gid, parcel) for gid in gids), key=lambda x: x["distance_m"])
                for name, gids in side_geoms.items()
            }
            left = side["left_finger"][0]
            right = side["right_finger"][0]
            # Fingertip center separation is measured from the closest contact
            # geom centers, not from the coupled joint coordinates.
            separation = float(np.linalg.norm(
                np.asarray(left["world_position_m"]) - np.asarray(right["world_position_m"])
            ))
            rows.append({
                "aperture_joint_m": float(aperture),
                "finger_joint_positions_m": [float(data.qpos[q]) for q in finger_qadr],
                "left": left,
                "right": right,
                "fingertip_separation_m": separation,
                "finger_symmetry_m": abs(left["distance_m"] - right["distance_m"]),
                "parcel_center_world_m": np.asarray(data.geom_xpos[parcel]).tolist(),
                "grasp_site_world_m": np.asarray(data.site_xpos[site]).tolist(),
                "parcel_relative_to_grasp_site_m": (np.asarray(data.geom_xpos[parcel]) - data.site_xpos[site]).tolist(),
                "parcel_axes_world": np.asarray(data.geom_xmat[parcel]).reshape(3, 3).tolist(),
                "parcel_face_normals_world": np.asarray(data.geom_xmat[parcel]).reshape(3, 3).T.tolist(),
                "parcel_widths_m": (2 * np.asarray(model.geom_size[parcel])).tolist(),
            })

        ap = np.asarray([r["aperture_joint_m"] for r in rows])
        left_d = np.asarray([r["left"]["distance_m"] for r in rows])
        right_d = np.asarray([r["right"]["distance_m"] for r in rows])
        left_zero, right_zero = _crossing(ap, left_d), _crossing(ap, right_d)
        left_extra = _extrapolated_crossing(ap, left_d)
        right_extra = _extrapolated_crossing(ap, right_d)
        valid = [
            r for r in rows
            if -0.002 <= r["left"]["distance_m"] <= 0
            and -0.002 <= r["right"]["distance_m"] <= 0
            and r["finger_symmetry_m"] <= 0.001
        ]
        adjacent = []
        for i in range(len(rows) - 1):
            if rows[i] in valid and rows[i + 1] in valid:
                adjacent.append([rows[i]["aperture_joint_m"], rows[i + 1]["aperture_joint_m"]])
        out = {
            "schema": "panda_aperture_geometry_analysis_v1",
            "scene": "centered_reset_diagnostic",
            "samples": len(rows),
            "finger_geometries": side_geoms,
            "rows": rows,
            "curves": {
                "left": {"minimum_distance_m": float(left_d.min()), "aperture_at_min_m": float(ap[np.argmin(left_d)]), "zero_crossing_aperture_m": left_zero, "extrapolated_zero_crossing_aperture_m": left_extra},
                "right": {"minimum_distance_m": float(right_d.min()), "aperture_at_min_m": float(ap[np.argmin(right_d)]), "zero_crossing_aperture_m": right_zero, "extrapolated_zero_crossing_aperture_m": right_extra},
                "zero_crossing_difference_m": None if left_extra is None or right_extra is None else abs(left_extra - right_extra),
            },
            "parcel": {
                "initial_center_world_m": initial_parcel.tolist(),
                "dimensions_m": (2 * np.asarray(model.geom_size[parcel])).tolist(),
                "grasp_axis_width_m": float(2 * model.geom_size[parcel][1]),
                "face_normals_are_columns_of_parcel_axes": True,
            },
            "pad_separation_m": {"minimum": float(min(r["fingertip_separation_m"] for r in rows)), "maximum": float(max(r["fingertip_separation_m"] for r in rows)), "closing_axis": "+Y in grasp-site frame"},
            "physics": {"timestep_s": float(model.opt.timestep), "solver": int(model.opt.solver), "parcel_mass_kg": float(model.body_mass[model.geom_bodyid[parcel]]), "parcel_has_free_joint": True, "mocap": False, "initialized_only_at_reset": True},
            "promotion": {"empty_no_finger_finger_contact": empty_finger_finger == 0, "empty_finger_finger_contacts": empty_finger_finger, "valid_samples": len(valid), "valid_adjacent_intervals": adjacent, "max_valid_penetration_m": -0.002, "max_valid_asymmetry_m": 0.001, "initial_overlap": bool(left_d[0] < 0 or right_d[0] < 0)},
            "decision": "no_nonempty_opposing_contact_interval" if not adjacent else "calibrated_dynamic_closure_candidate",
        }
        return out
    finally:
        scene.unlink(missing_ok=True)


if __name__ == "__main__":
    result = run()
    path = ROOT / "artifacts/panda_aperture_sweep_analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(path), "decision": result["decision"], "curves": result["curves"], "pad_separation_m": result["pad_separation_m"], "promotion": result["promotion"]}, indent=2))
