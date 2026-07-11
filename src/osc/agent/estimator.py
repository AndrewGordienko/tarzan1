"""StateEstimator: percept history -> BeliefState.

Responsibilities the agent used to get for free from SimState, now earned:
  * data association -- match nameless detections to persistent track IDs across
    frames by nearest-neighbour gating around each track's VELOCITY-PREDICTED
    next pose (so a fast, smoothly-moving carried object is not lost, while an
    object carried above another stays distinct in 3D),
  * coasting -- when a track is occluded/dropped, keep coasting on its last
    velocity and grow its position uncertainty until re-observed,
  * grasp inference -- decide what (if anything) is held from gripper closure +
    an object co-moving with the gripper; ground-truth grasp is never available,
  * appearance/size estimation for later correspondence.
"""
from __future__ import annotations

import numpy as np

from ..geometry import dist_xy, dist_xyz
from ..perception.detections import Percept
from .belief import BeliefObject, BeliefState

GATE = 0.06          # gating radius around the predicted pose (3D, metres)
GRASP_XY = 0.035
GRASP_Z = 0.04
SIZE_PRIOR_VAR = 0.02 ** 2   # initial size variance before any fusion
SIZE_MEAS_VAR = 0.010 ** 2   # assumed per-observation size measurement variance
# 99% chi-square gate with three attributes.  The gate is applied only to axes
# the current camera can actually measure, so an occluded-axis pseudo-prior
# cannot manufacture either a rejection or evidence.
SIZE_NIS_GATE = 11.34
OCCLUDED_MEAS_STD = 0.5
ASSOCIATION_COV_INFLATION = 4.0
ASSOCIATION_RECOVERY_FRAMES = 2


class StateEstimator:
    def __init__(self):
        self.tracks: dict[str, BeliefObject] = {}
        self.vel: dict[str, np.ndarray] = {}
        self.svar: dict[str, float] = {}       # per-track size variance (Kalman)
        self._next = 0
        self.grasped: str | None = None
        self.t = 0
        self.association_events: list[dict] = []  # audit-only, agent-visible innovation outcomes

    def _new_track(self, det, t) -> str:
        tid = f"t{self._next}"; self._next += 1
        # An occluded axis is a prior-like placeholder, not a precise first
        # observation.  Its covariance must carry that fact into correspondence.
        init_var = np.full(3, SIZE_PRIOR_VAR)
        if det.size_meas_std is not None:
            init_var = np.maximum(init_var, det.size_meas_std ** 2)
        self.tracks[tid] = BeliefObject(
            track_id=tid, pose=det.pose.copy(), size=det.size.copy(),
            shape=det.shape, color=det.color, marker=det.marker, pos_std=0.02,
            size_std=SIZE_PRIOR_VAR ** 0.5,
            size_var=init_var.copy(), last_seen=t, visible=True)
        self.vel[tid] = np.zeros(4)
        self.svar[tid] = init_var.copy()              # per-axis size variance
        return tid

    def update(self, percept: Percept) -> BeliefState:
        t = percept.t
        for o in self.tracks.values():
            o.visible = False
        predicted = {tid: self.tracks[tid].pose + self.vel[tid] for tid in self.tracks}
        unmatched = set(self.tracks)
        dets = list(percept.detections)

        # 1) grasp first: decide the held track, pin it to the gripper, and REMOVE
        #    its detection from the pool so a fast-moving held object cannot spawn
        #    ghost tracks along the transport path.
        held = self._infer_grasp(percept)
        if held is not None:
            # consume the held object's detection by 3D distance: the held object
            # sits AT the gripper, whereas a support being stacked onto is BELOW
            # it -- using xy alone would wrongly eat the support during descent.
            if dets:
                gi = min(range(len(dets)), key=lambda i: dist_xyz(dets[i].pose, percept.gripper))
                if dist_xyz(dets[gi].pose, percept.gripper) < 0.045:
                    dets.pop(gi)
            o = self.tracks[held]
            self.vel[held] = percept.gripper - o.pose
            o.pose = o.pose.copy(); o.pose[:3] = percept.gripper[:3]
            o.pos_std = min(o.pos_std, 0.02); o.visible = True; o.last_seen = t
            unmatched.discard(held)

        # 2) associate remaining detections to remaining tracks
        for det in dets:
            best, best_d = None, GATE
            for tid in unmatched:
                d = dist_xyz(predicted[tid], det.pose)
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                self._new_track(det, t)
            else:
                o = self.tracks[best]
                new_pose = 0.4 * predicted[best] + 0.6 * det.pose
                self.vel[best] = 0.5 * self.vel[best] + 0.5 * (new_pose - o.pose)
                o.pose = new_pose
                # Attribute fusion is only valid after identity-safe association.
                # A spatial nearest-neighbour match can be wrong when objects cross
                # or a track is reacquired.  Test the size innovation first; a
                # rejected measurement never gets averaged into the track.
                # Reacquisition/duplicate-track cleanup is not a new independent
                # attribute sample, so its first observation is deliberately not
                # fused either.
                consecutive = (t == o.last_seen + 1)
                if consecutive:
                    self._fuse_size_if_consistent(best, det)
                o.shape, o.color = det.shape, det.color
                if det.marker != "unknown":       # do not erase side evidence on return to top
                    o.marker = det.marker
                o.pos_std = max(0.006, o.pos_std * 0.6)
                o.last_seen = t; o.visible = True
                unmatched.discard(best)

        for tid in unmatched:                     # coast occluded tracks
            o = self.tracks[tid]
            o.pose = o.pose + self.vel[tid]
            self.vel[tid] *= 0.7
            o.pos_std = min(0.15, o.pos_std + 0.01)

        self._merge_duplicates()
        self.t = t
        return self._snapshot(percept)

    def _fuse_size_if_consistent(self, tid: str, det) -> bool:
        """Innovation-gated Kalman update for object attributes.

        The trajectory association is intentionally separate from this guard:
        when it looks like a different object's detection entered a track we keep
        its pose association available for subsequent recovery, but do not let
        repeated wrong-object frames drive the size covariance to zero.
        """
        o = self.tracks[tid]
        sv = self.svar[tid]
        mv = (det.size_meas_std ** 2 if det.size_meas_std is not None
              else np.full(3, SIZE_MEAS_VAR))
        informative = mv < OCCLUDED_MEAS_STD ** 2
        if not np.any(informative):
            return False
        innovation = det.size[informative] - o.size[informative]
        # Inflation communicates uncertainty downstream; it must not silently
        # widen the identity gate enough for the repeatedly wrong object to
        # become "consistent" on the next frame.  While contested, compare to a
        # bounded pre-change attribute uncertainty until stable evidence returns.
        gate_sv = (np.minimum(sv, SIZE_PRIOR_VAR) if o.association_contested else sv)
        innovation_cov = gate_sv[informative] + mv[informative]
        nis = float(np.sum((innovation ** 2) / np.maximum(innovation_cov, 1e-12)))
        if nis > SIZE_NIS_GATE:
            # Suspected identity change: retain the old mean, make it honestly
            # uncertain again, and surface the contest to the resolution layer.
            self.svar[tid] = np.maximum(sv * ASSOCIATION_COV_INFLATION,
                                        np.full(3, SIZE_PRIOR_VAR))
            o.association_contested = True
            o.association_stable_frames = 0
            o.size_var = self.svar[tid].copy()
            o.size_std = float(np.sqrt(self.svar[tid].max()))
            self.association_events.append(dict(t=self.t, track_id=tid, nis=nis, accepted=False))
            return False

        K = sv[informative] / (sv[informative] + mv[informative])
        o.size[informative] = o.size[informative] + K * innovation
        self.svar[tid][informative] = (1.0 - K) * sv[informative]
        if o.association_contested:
            o.association_stable_frames += 1
            if o.association_stable_frames >= ASSOCIATION_RECOVERY_FRAMES:
                o.association_contested = False
        o.size_var = self.svar[tid].copy()
        o.size_std = float(np.sqrt(self.svar[tid].max()))
        self.association_events.append(dict(t=self.t, track_id=tid, nis=nis, accepted=True))
        return True

    def _merge_duplicates(self, dist=0.025):
        """Two tracks at nearly the same 3D point are the same object: drop the
        less-trustworthy one. Keeps a grasped or lower-index track. Removes the
        transient duplicates that fast/occluded motion can spawn."""
        ids = list(self.tracks)
        drop = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a in drop or b in drop:
                    continue
                if dist_xyz(self.tracks[a].pose, self.tracks[b].pose) < dist:
                    keep, lose = self._prefer(a, b)
                    drop.add(lose)
        for tid in drop:
            self.tracks.pop(tid, None); self.vel.pop(tid, None); self.svar.pop(tid, None)

    def _prefer(self, a, b):
        if self.grasped == a:
            return a, b
        if self.grasped == b:
            return b, a
        # keep the more recently/precisely observed, tie-break to lower index
        ka = (self.tracks[a].pos_std, int(a[1:]))
        kb = (self.tracks[b].pos_std, int(b[1:]))
        return (a, b) if ka <= kb else (b, a)

    def _infer_grasp(self, percept: Percept) -> str | None:
        if percept.gripper_closed < 0.6:
            self.grasped = None
            return None
        if self.grasped is not None and self.grasped in self.tracks:
            return self.grasped
        best, best_d = None, GRASP_XY
        for tid, o in self.tracks.items():
            if dist_xy(o.pose, percept.gripper) <= best_d and abs(o.pose[2] - percept.gripper[2]) <= GRASP_Z:
                best, best_d = tid, dist_xy(o.pose, percept.gripper)
        self.grasped = best
        return best

    def _snapshot(self, percept: Percept) -> BeliefState:
        b = BeliefState(
            objects={k: v.copy() for k, v in self.tracks.items()},
            gripper=percept.gripper.copy(), gripper_closed=percept.gripper_closed,
            grasped=self.grasped, t=percept.t)
        if self.grasped is not None and self.grasped in self.tracks:
            b.grasp_confidence = 1.0 - min(1.0, self.tracks[self.grasped].pos_std / 0.1)
        return b
