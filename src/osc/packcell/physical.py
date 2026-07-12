from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class PhaseResult:
    phase: str
    success: bool
    reason: str = ""
    contacts: int = 0
    force_n: float = 0.0
    steps: int = 0


class PackCell:
    """One-object contact-valid upper-bound interface.

    The agent receives only RGB-D/segmentation and controller state. Privileged
    object pose is available through ``scorer_state`` exclusively. After reset,
    this class writes only actuator controls; MuJoCo advances object qpos/qvel.
    """

    def __init__(self, seed: int = 0, width: int = 640, height: int = 480):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install osc[embodied] for PackCell") from exc
        self.mujoco = mujoco
        self.width, self.height, self.seed = width, height, seed
        self.model = self.data = self.renderer = None
        self.object_body = self.object_qadr = None
        self._reset_done = False

    def reset(self):
        m = self.mujoco
        tiny = Path(__file__).resolve().parents[4] / "tinyvla" / "SO-ARM100" / "Simulation" / "SO101"
        source = tiny / "task.xml"
        if not source.exists():
            raise RuntimeError(f"TinyVLA SO-101 assets not found at {source}")
        xml = source.read_text().replace('contype="2" conaffinity="2"', 'contype="1" conaffinity="1"')
        calibration = tiny / "so101_new_calib.xml"
        cal_runtime = tiny / "packcell_calibration_runtime.xml"
        cal_xml = calibration.read_text().replace('<site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 -0 0.707107 -2.37788e-17"/>',
                          '<site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 -0 0.707107 -2.37788e-17"/><site name="grasp_site" pos="0.01644 0.000055 -0.03093"/>')
        cal_runtime.write_text(cal_xml)
        xml = xml.replace('file="so101_new_calib.xml"', f'file="{cal_runtime.name}"')
        xml = xml.replace('<global azimuth="160" elevation="-20" offwidth="640" offheight="480"/>',
                          '<global azimuth="160" elevation="-20" offwidth="640" offheight="480"/>')
        generated = tiny / "packcell_v1_runtime.xml"
        generated.write_text(xml)
        try:
            self.model = m.MjModel.from_xml_path(str(generated))
        finally:
            generated.unlink(missing_ok=True)
            cal_runtime.unlink(missing_ok=True)
        self.data = m.MjData(self.model)
        self.renderer = m.Renderer(self.model, height=self.height, width=self.width)
        self.object_body = m.mj_name2id(self.model, m.mjtObj.mjOBJ_BODY, "cube_red")
        jid = m.mj_name2id(self.model, m.mjtObj.mjOBJ_JOINT, "cube_red_free")
        self.object_qadr = int(self.model.jnt_qposadr[jid])
        # Reset-time initialization is the only object-state write in this module.
        m.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = [0, -1.2, .6, 1.2, 0, 1.2]
        self.data.qpos[self.object_qadr:self.object_qadr + 3] = [.20, -.06, .087]
        self.data.qpos[self.object_qadr + 3:self.object_qadr + 7] = [1, 0, 0, 0]
        m.mj_forward(self.model, self.data)
        self._reset_done = True
        return self.agent_observation()

    def agent_observation(self) -> dict[str, Any]:
        if not self._reset_done: raise RuntimeError("reset first")
        self.renderer.update_scene(self.data, camera="front")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering(); depth = self.renderer.render().copy(); self.renderer.disable_depth_rendering()
        self.renderer.enable_segmentation_rendering(); seg = self.renderer.render().copy(); self.renderer.disable_segmentation_rendering()
        gid = self.model.geom("cube_red").id
        return {"rgb": rgb, "depth": depth, "mask": seg[:, :, 1] == gid,
                "controller": self.data.qpos[:6].copy(),
                "contacts": self._contact_observation()}

    def controller_state(self) -> dict[str, Any]:
        return {"joint_position": self.data.qpos[:6].copy(), "joint_velocity": self.data.qvel[:6].copy(),
                "actuator_control": self.data.ctrl.copy()}

    def ee_position(self):
        sid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        return self.data.site_xpos[sid].copy()

    def actuator_ranges(self):
        return self.model.actuator_ctrlrange.copy()

    def contact_summary(self):
        return self._contact_observation()

    def scorer_state(self) -> dict[str, Any]:
        """Privileged evaluation view; never passed to controller methods."""
        a = self.object_qadr
        return {"object_position": self.data.qpos[a:a + 3].copy(),
                "object_velocity": self.data.qvel[self.model.jnt_dofadr[self.model.body_jntadr[self.object_body]]:][:3].copy(),
                "object_body": self.object_body}

    def _contact_observation(self):
        out = []
        for i, c in enumerate(self.data.contact):
            force = np.zeros(6); self.mujoco.mj_contactForce(self.model, self.data, i, force)
            out.append({"geom1": int(c.geom1), "geom2": int(c.geom2), "normal_force_n": float(abs(force[0]))})
        return tuple(out)

    def step_control(self, action: np.ndarray) -> dict[str, Any]:
        if not self._reset_done: raise RuntimeError("reset first")
        self.data.ctrl[:] = np.asarray(action, dtype=float)  # controller state only
        self.mujoco.mj_step(self.model, self.data)
        return self.agent_observation()

    def verify(self) -> PhaseResult:
        s = self.scorer_state(); p = s["object_position"]; v = np.linalg.norm(s["object_velocity"])
        inside = .095 < p[0] < .165 and -.11 < p[1] < -.04 and .075 < p[2] < .12
        return PhaseResult("verify", bool(inside and v < .03), "inside_supported_stationary" if inside and v < .03 else "not_inside_or_moving", len(self.data.contact), 0.0)


def run_packcell_benchmark(seeds=range(20)):
    rows = []
    for seed in seeds:
        try:
            cell = PackCell(seed); cell.reset()
            result = cell.verify()
            rows.append({"seed": seed, "success": False, "phase": asdict(result), "failure_attribution": "planning"})
        except RuntimeError as exc:
            rows.append({"seed": seed, "success": False, "failure_attribution": "environment", "reason": str(exc)})
    return {"episodes": len(rows), "successes": sum(r["success"] for r in rows), "rows": rows,
            "status": "upper_bound_controller_not_yet_connected"}
