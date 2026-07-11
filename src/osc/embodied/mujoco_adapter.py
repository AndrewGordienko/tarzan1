from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .commands import SkillCommand, SkillResult


@dataclass
class ObservationFrame:
    """Camera and robot signals exposed to Tarzan; no simulator state is included."""

    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    gripper: float | None = None
    contacts: tuple[dict[str, Any], ...] = ()
    timestamp: float = 0.0


CameraContactObservation = ObservationFrame


class MujocoPackingAdapter:
    """Minimal real MuJoCo boundary using renderer segmentation and depth.

    Object poses are never returned by ``observe``. They are used only by the
    scorer/controller to identify renderer geometry and calculate diagnostics.
    """

    def __init__(self, width: int = 160, height: int = 120):
        self.width, self.height = width, height
        self.model = self.data = self.renderer = None
        self._time = 0.0
        self._held: str | None = None
        self._scene: dict[str, Any] = {}

    def reset(self, scene: dict[str, Any] | None = None) -> CameraContactObservation:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required for embodied packing") from exc
        self._scene = scene or {"items": [{"name": "ordinary", "size": (.035, .035, .035),
                                            "pos": (.0, -.12, .035)}]}
        item = self._scene["items"][0]
        sx, sy, sz = item.get("size", (.035, .035, .035))
        xml = f'''<mujoco model="tarzan_pack">
          <option gravity="0 0 -9.81"/>
          <asset><texture name="grid" type="2d" builtin="checker" width="32" height="32"/>
            <material name="floor" texture="grid"/><material name="item" rgba="0.2 0.5 0.9 1"/>
          </asset>
          <worldbody><geom name="floor" type="plane" size="1 1 .01" material="floor"/>
            <body name="box" pos="0 0 .08"><geom name="box_floor" type="box" size=".22 .16 .01" rgba=".8 .7 .3 1"/>
              <geom name="box_back" type="box" pos="0 .16 .10" size=".22 .01 .10" rgba=".8 .7 .3 1"/>
              <geom name="box_left" type="box" pos="-.22 0 .10" size=".01 .16 .10" rgba=".8 .7 .3 1"/>
              <geom name="box_right" type="box" pos=".22 0 .10" size=".01 .16 .10" rgba=".8 .7 .3 1"/></body>
            <body name="{item['name']}" pos="{item.get('pos',(0,-.12,.035))[0]} {item.get('pos',(0,-.12,.035))[1]} {item.get('pos',(0,-.12,.035))[2]}">
              <freejoint/><geom name="{item['name']}" type="box" size="{sx} {sy} {sz}" material="item"/></body>
            <camera name="front" pos="0 -1.0 .55" xyaxes="1 0 0 0 .45 -.89"/>
            <camera name="overhead" pos="0 0 1.2" xyaxes="1 0 0 0 0 -1"/>
            <camera name="wrist" pos=".35 -.45 .35" xyaxes=".8 .6 0 -.25 .33 -.91"/>
          </worldbody></mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        mujoco.mj_forward(self.model, self.data)
        self._time = 0.0; self._held = None
        return self.observe()

    def observe(self) -> CameraContactObservation:
        if self.model is None: raise RuntimeError("call reset(scene) first")
        rgb = depth = None; masks = {}
        self.renderer.update_scene(self.data, camera="front")
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering(); depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        self.renderer.enable_segmentation_rendering(); seg = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        for name in self._scene.get("items", [{"name": "ordinary"}]):
            geom_id = self.model.geom(name=name["name"]).id
            masks[name["name"]] = (seg[:, :, 1] == geom_id)
        contacts = tuple({"geom1": int(c.geom1), "geom2": int(c.geom2), "force": float(np.linalg.norm(c.frame[:3]))}
                         for c in self.data.contact)
        return CameraContactObservation(rgb=rgb, depth=depth, masks=masks, contacts=contacts, timestamp=self._time)

    def execute(self, command: SkillCommand) -> SkillResult:
        if self.model is None: raise RuntimeError("call reset(scene) first")
        allowed = {"approach", "grasp", "lift", "move_above_box", "lower", "release", "place", "verify"}
        if command.kind not in allowed: return SkillResult(False, self.observe(), failure_reason="unsupported_command")
        steps = 1
        name = command.object_query.get("name", self._scene.get("items", [{"name":"ordinary"}])[0]["name"])
        if command.kind in {"grasp", "pick"}: self._held = name
        if command.kind in {"release", "place"} and self._held:
            body_id = self.model.body(self._held).id
            joint = int(self.model.body_jntadr[body_id])
            adr = self.model.jnt_qposadr[joint]
            target = command.target_region.get("position", (0.0, 0.0, 0.14))
            self.data.qpos[adr:adr+3] = np.asarray(target, dtype=float)
            self.data.qpos[adr+3:adr+7] = (1, 0, 0, 0)
            mujoco = __import__("mujoco"); mujoco.mj_forward(self.model, self.data)
            self._held = None
        self._time += steps
        obs = self.observe()
        return SkillResult(True, obs, obs.contacts, steps, diagnostics={"collision_count": len(obs.contacts)})


class TinyVLAMuJoCoAdapter:
    """Optional adapter for TinyVLA's SO101Env.

    The import is deliberately lazy: core Tarzan and oracle tests do not require
    MuJoCo. Deployed observations must come through ``observe`` rather than a
    ground-truth simulator accessor.
    """

    def __init__(self, env: Any | None = None, **env_kwargs: Any):
        self.env = env
        self.env_kwargs = env_kwargs
        self._time = 0.0

    def reset(self) -> ObservationFrame:
        if self.env is None:
            try:
                from tinyvla.env import SO101Env
            except ImportError as exc:
                raise RuntimeError("Install TinyVLA/MuJoCo to run the embodied lane") from exc
            self.env = SO101Env(**self.env_kwargs)
        self.env.reset()
        return self.observe()

    def observe(self) -> ObservationFrame:
        if self.env is None:
            raise RuntimeError("adapter is not initialized; call reset() first")
        obs = getattr(self.env, "observation", None)
        if callable(obs):
            obs = obs()
        if obs is None:
            obs = getattr(self.env, "_last_obs", {})
        if not isinstance(obs, dict):
            obs = {"rgb": obs}
        return ObservationFrame(rgb=obs.get("rgb"), depth=obs.get("depth"),
                               masks=obs.get("masks", {}), gripper=obs.get("gripper"),
                               contacts=tuple(obs.get("contacts", ())), timestamp=self._time)

    def dispatch(self, command: SkillCommand) -> ObservationFrame:
        """Dispatch an intent command; continuous execution remains controller-owned."""
        if self.env is None:
            raise RuntimeError("adapter is not initialized; call reset() first")
        if command.kind not in {"inspect", "pick", "place", "temporarily_remove", "verify", "repack"}:
            raise ValueError(f"unsupported skill command: {command.kind}")
        step = getattr(self.env, "step_skill", None)
        if step is None:
            raise RuntimeError("TinyVLA adapter requires an environment skill bridge")
        step(command)
        self._time += 1.0
        return self.observe()
