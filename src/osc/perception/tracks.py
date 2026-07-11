"""Stage-A front-end over BELIEF: track trajectories, contact, keyframes.

Consumes the sequence of BeliefStates the estimator produced while watching the
one demonstration (so it is subject to the same perception limits as execution).
Tracks are keyed by anonymous estimator IDs; roles are inferred downstream by
function, never from names.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose
from ..agent.belief import BeliefState


@dataclass
class DemoTrace:
    object_tracks: dict[str, list[Pose]]
    gripper: list[Pose]
    gripper_closed: list[float]
    grasped: list                       # track_id or None, per frame
    features: dict[str, np.ndarray]     # track_id -> appearance/geometry signature
    feature_vars: dict[str, np.ndarray] # track_id -> uncertainty of that signature
    T: int = 0

    def contact_of(self, tid: str, t: int) -> bool:
        return self.grasped[t] == tid


def extract_tracks(beliefs: list[BeliefState]) -> DemoTrace:
    # union of track ids seen across the demo (estimator keeps them stable)
    ids = []
    for b in beliefs:
        for k in b.objects:
            if k not in ids:
                ids.append(k)
    tracks = {i: [] for i in ids}
    feats, feat_vars = {}, {}
    grip, gclosed, grasped = [], [], []
    for b in beliefs:
        for i in ids:
            if i in b.objects:
                tracks[i].append(b.objects[i].pose.copy())
                feats[i] = b.objects[i].feature()
                feat_vars[i] = b.objects[i].feature_var()
            else:
                tracks[i].append(tracks[i][-1] if tracks[i] else np.zeros(4))
        grip.append(b.gripper.copy())
        gclosed.append(b.gripper_closed)
        grasped.append(b.grasped)
    return DemoTrace(tracks, grip, gclosed, grasped, feats, feat_vars, T=len(beliefs))


@dataclass
class Keyframe:
    t: int
    reason: str
    grasped: str | None
    object_poses: dict[str, Pose] = field(default_factory=dict)
    gripper: Pose | None = None


def segment_keyframes(trace: DemoTrace, still_thresh: float = 0.004) -> list[Keyframe]:
    kfs: list[Keyframe] = []

    def push(t, reason):
        kfs.append(Keyframe(t=t, reason=reason, grasped=trace.grasped[t],
                            object_poses={i: trace.object_tracks[i][t] for i in trace.object_tracks},
                            gripper=trace.gripper[t]))

    push(0, "start")
    prev_g = trace.grasped[0]
    for t in range(1, trace.T):
        if trace.grasped[t] != prev_g:
            push(t, "contact-change")
            prev_g = trace.grasped[t]
            continue
        prev_g = trace.grasped[t]
        if trace.grasped[t] is None:      # contact-free push/wipe boundary only
            if _max_speed(trace, t - 1) > still_thresh and _max_speed(trace, t) <= still_thresh:
                push(t, "motion-still")
    if kfs[-1].t != trace.T - 1:
        push(trace.T - 1, "end")
    return _dedup(kfs)


def _max_speed(trace: DemoTrace, t: int) -> float:
    if t <= 0:
        return 0.0
    return max(float(np.linalg.norm(trace.object_tracks[i][t][:3] - trace.object_tracks[i][t - 1][:3]))
               for i in trace.object_tracks)


def _dedup(kfs):
    out = []
    for k in kfs:
        if out and out[-1].t == k.t:
            continue
        out.append(k)
    return out
