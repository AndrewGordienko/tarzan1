"""ToyTabletopSim -- a small NumPy rigid-body tabletop that runs on CPU.

It is intentionally not a physics engine. It captures exactly the phenomena the
benchmark stresses and nothing more:

  * a TCP that moves toward a target with first-order actuator delay,
  * grasping when the gripper closes near an object,
  * gravity: an ungrasped object settles onto the table or onto a support
    object it overlaps (stacking),
  * friction/mass modulating how far a pushed object slides,
  * collisions (TCP or held object overlapping another object),
  * force violations (commanding into a heavy blocked object),
  * irreversibility (an object pushed past the table edge falls and is lost).

The point is a fully runnable A->D loop on a laptop today; ManiSkillBackend
swaps in for photoreal GPU sim later behind the same SimBackend interface.
"""
from __future__ import annotations

import numpy as np

from ..geometry import Pose, dist_xy, dist_xyz, pose
from .base import Action, Observation, SimObject, SimState, StepInfo

GRASP_XY = 0.03      # max planar offset to acquire a grasp
GRASP_Z = 0.03       # max vertical offset to acquire a grasp
CLOSE_THRESH = 0.6   # gripper_closed above this counts as "closed"
FORCE_LIMIT = 8.0    # abstract force units before a violation is flagged

# Per-camera size measurement std, per axis [x, y, z]. A view cannot resolve a
# dimension along its optical axis: the default TOP view foreshortens HEIGHT (z);
# a FRONT/SIDE view reveals it but blurs the axis it looks down. This is the
# explicit partial-observability that a change_viewpoint action must overcome.
# (A per-axis abstraction of camera geometry; real SE3 calibration arrives with
# the ManiSkill backend.) The same cameras exist in demo and deployment.
# A dimension along a view's optical axis is OCCLUDED (uninformative, std OCCLUDED)
# -- not merely noisy -- so repeated frames from that view carry NO new evidence.
# The sharp axes get a small std. Same cameras exist in demo and deployment.
_SHARP, OCCLUDED = 0.002, 1.0
NEUTRAL_SIZE = 0.040                     # uninformed prior emitted on an occluded axis
CAMERA_MEAS_STD = {
    "top":     (_SHARP, _SHARP, OCCLUDED),   # default: footprint sharp, height occluded
    "top_rot": (_SHARP, _SHARP, OCCLUDED),   # a DIFFERENT view that still can't see height
    "front":   (_SHARP, OCCLUDED, _SHARP),   # reveals height (z); occludes depth (y)
    "side":    (OCCLUDED, _SHARP, _SHARP),   # reveals height (z); occludes width (x)
}
# Labels are printed on the y-facing side in this toy camera model.  They are
# deliberately unavailable from TOP and FRONT, so a side observation is a real
# new appearance channel rather than a name/order shortcut.
CAMERA_MARKER_VISIBLE = {"top": False, "top_rot": False, "front": False, "side": True}


class ToyTabletopSim:
    def __init__(self, actuator_delay: float = 0.25, lighting: float = 0.5,
                 camera_jitter: float = 0.0, rng: np.random.Generator | None = None,
                 camera_model: bool = False):
        # actuator_delay in [0,1): fraction of the remaining error left uncorrected
        # each step (0 = instantaneous, ->1 = very sluggish).
        self.actuator_delay = float(np.clip(actuator_delay, 0.0, 0.95))
        self.lighting = lighting
        self.camera_jitter = camera_jitter
        self.rng = rng or np.random.default_rng(0)
        self._s: SimState | None = None
        self._pre_step_hook = None       # used by disturbance injection
        self.camera_model = camera_model # off by default -> exact sizes (headline bench)
        self.camera = "top"              # current viewpoint (observation state)

    def set_camera(self, name: str) -> None:
        """Move the camera. An ACTION, not privileged state -- changes which size
        axis is observable in subsequent percepts."""
        if name in CAMERA_MEAS_STD:
            self.camera = name

    # -- SimBackend -------------------------------------------------------
    def reset(self, state: SimState) -> Observation:
        self._s = state.copy()
        self._settle()
        return self.observe()

    def state(self) -> SimState:
        return self._s.copy()

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        s, info = self._s, StepInfo()
        if self._pre_step_hook is not None:
            self._pre_step_hook(s, info)

        # 1) actuator dynamics: TCP eases toward target (first-order lag).
        alpha = 1.0 - self.actuator_delay
        new_g = s.gripper + alpha * (action.target - s.gripper)
        move = new_g - s.gripper
        s.gripper = new_g
        s.gripper_closed = float(np.clip(action.gripper_close, 0.0, 1.0))

        # 2) grasp acquisition / release.
        closed = s.gripper_closed >= CLOSE_THRESH
        if s.grasped is None and closed:
            for name, o in s.objects.items():
                if name in s.fallen:
                    continue
                if dist_xy(s.gripper, o.pose) <= GRASP_XY and abs(s.gripper[2] - o.pose[2]) <= GRASP_Z:
                    s.grasped = name
                    break
        elif s.grasped is not None and not closed:
            s.grasped = None

        # 3) carry grasped object; push/collide with the rest.
        if s.grasped is not None:
            held = s.objects[s.grasped]
            held.pose = pose(s.gripper[0], s.gripper[1], s.gripper[2], s.gripper[3])

        self._resolve_contacts(move, info)
        self._settle(info)
        s.t += 1
        return self.observe(), info

    # -- internals --------------------------------------------------------
    def _resolve_contacts(self, move: np.ndarray, info: StepInfo) -> None:
        """Lateral contact only. A top-down grasp approach or a descent to stack
        (near-zero horizontal motion, or large vertical offset) must NOT push the
        object -- otherwise you could never grasp or stack. Pushing models a
        side-swipe: the mover translates horizontally into an object at a similar
        height."""
        s = self._s
        mover_name = s.grasped
        mover = s.objects[mover_name] if mover_name else None
        ref = mover.pose if mover is not None else s.gripper
        speed_xy = float(np.hypot(move[0], move[1]))
        if speed_xy < 1e-3:                     # vertical / stationary: no push
            return
        for name, o in s.objects.items():
            if name == mover_name or name in s.fallen:
                continue
            clear = _radius(o) + (_radius(mover) if mover is not None else 0.015)
            same_height = abs(ref[2] - o.pose[2]) < (_radius(o) + o.size[2] / 2)
            if same_height and dist_xy(ref, o.pose) < clear:
                info.collision = True
                # push the struck object; heavier/higher-friction slides less.
                resist = o.mass * (1.0 + o.friction)
                push = max(0.0, speed_xy - 0.002) / (1.0 + resist)
                # contact reaction force ~ commanded speed into the object scaled
                # by how much it resists. Fires on ANY hard collision -- the old
                # `push < 1e-4` gate made this unreachable for every configured
                # mass (min push ~0.004), so the zero rate was structural, not
                # earned. A gentle agent (speed<0.02) still stays well under the
                # limit; only driving hard into a heavy object violates.
                if speed_xy * (1.0 + resist) * 40.0 > FORCE_LIMIT:
                    info.force_violation = True
                direction = (o.pose[:2] - ref[:2])
                n = np.linalg.norm(direction)
                if n > 1e-6:
                    o.pose[:2] += (direction / n) * push
                    info.events.append(f"pushed:{name}")

    def _settle(self, info: "StepInfo | None" = None) -> None:
        """Apply gravity/support: unheld objects rest on table or on a support;
        objects beyond the table edge fall off (irreversible)."""
        s = self._s
        xmin, xmax, ymin, ymax = s.table_bounds
        for name, o in s.objects.items():
            if name == s.grasped or name in s.fallen:
                continue
            if not (xmin <= o.pose[0] <= xmax and ymin <= o.pose[1] <= ymax):
                s.fallen.add(name)            # newly left the table this step
                o.pose[2] = s.table_z - 0.5
                if info is not None:
                    info.irreversible = True
                    info.events.append(f"fell:{name}")
                continue
            support_top = s.table_z
            for other, oo in s.objects.items():
                if other in (name, s.grasped) or other in s.fallen:
                    continue
                # only an object strictly below can be a support -- otherwise two
                # co-located objects would each climb the other every step.
                if oo.pose[2] >= o.pose[2]:
                    continue
                if dist_xy(o.pose, oo.pose) < (_radius(o) + _radius(oo)) * 0.7:
                    # top surface of the support is its centre + HALF its height.
                    support_top = max(support_top, oo.pose[2] + oo.size[2] / 2)
            o.pose[2] = support_top + o.size[2] / 2

    def observe(self) -> Observation:
        s = self._s
        # perception noise grows as lighting worsens (lighting 1=bright,0=dark).
        noise = 0.0015 + 0.01 * (1.0 - float(self.lighting)) + 0.5 * self.camera_jitter
        tracks, contact = {}, {}
        for name, o in s.objects.items():
            p = o.pose.copy()
            if name not in s.fallen:
                p[:2] += self.rng.normal(0, noise, size=2)
            tracks[name] = p
            contact[name] = (name == s.grasped)
        return Observation(object_tracks=tracks, gripper=s.gripper.copy(),
                           gripper_closed=s.gripper_closed, contact=contact, t=s.t)

    def perceive(self) -> "Percept":
        """Sensor model: ground truth -> an UNORDERED list of nameless detections
        with lighting-dependent position noise. This is the only thing the agent
        is allowed to see (further corrupted by Corruptor in AgentEnv). Names,
        ground-truth grasp, mass and friction are NOT exposed."""
        from ..perception.detections import Detection, Percept
        s = self._s
        noise = 0.0015 + 0.01 * (1.0 - float(self.lighting)) + 0.5 * self.camera_jitter
        dets = []
        for name, o in s.objects.items():
            if name in s.fallen:
                continue
            p = o.pose.copy()
            p[:2] += self.rng.normal(0, noise, size=2)
            if self.camera_model:
                # occluded axes emit an uninformed prior (no per-frame evidence);
                # sharp axes emit the true size + small calibrated noise.
                meas = np.array(CAMERA_MEAS_STD[self.camera], dtype=float)
                occ = meas >= OCCLUDED
                sz = np.where(occ, NEUTRAL_SIZE, o.size + self.rng.normal(0, 1.0, size=3) * meas)
                marker = o.marker if CAMERA_MARKER_VISIBLE[self.camera] else "unknown"
                dets.append(Detection(pose=p, size=sz, shape=o.shape, color=o.color, marker=marker,
                                      contact=(name == s.grasped), size_meas_std=meas.copy()))
            else:
                dets.append(Detection(pose=p, size=o.size.copy(), shape=o.shape,
                                      color=o.color, marker=o.marker, contact=(name == s.grasped)))
        self.rng.shuffle(dets)
        return Percept(detections=dets, gripper=s.gripper.copy(),
                       gripper_closed=s.gripper_closed, t=s.t)


def _radius(o: SimObject) -> float:
    return float(max(o.size[0], o.size[1]) / 2)
