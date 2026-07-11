"""Stage-A front-end: raw observations -> object tracks, contact states, keyframes.

In the full system this is a trained video encoder (V-JEPA-2 / IMOP-style) that
emits per-object tracks, contact events and keyframes from pixels. On the toy
backend we get privileged (noisy) tracks directly from Observations; the *logic*
that turns tracks+contacts into keyframes is identical to what would sit on top
of a real encoder, so this module is the honest interface, with the encoder
swapped in later behind `extract_tracks`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry import Pose
from ..sim.base import Observation


@dataclass
class DemoTrace:
    """The single demonstration, as perception sees it: a time series of
    per-object poses plus the gripper's contact/closure signal."""
    object_tracks: dict[str, list[Pose]]         # name -> [pose per frame]
    gripper: list[Pose]
    gripper_closed: list[float]
    contact: dict[str, list[bool]]               # name -> [in-contact per frame]
    T: int = 0


def extract_tracks(observations: list[Observation]) -> DemoTrace:
    names = list(observations[0].object_tracks.keys())
    trace = DemoTrace(object_tracks={n: [] for n in names}, gripper=[],
                      gripper_closed=[], contact={n: [] for n in names})
    for o in observations:
        for n in names:
            trace.object_tracks[n].append(o.object_tracks[n])
            trace.contact[n].append(o.contact[n])
        trace.gripper.append(o.gripper)
        trace.gripper_closed.append(o.gripper_closed)
    trace.T = len(observations)
    return trace


@dataclass
class Keyframe:
    t: int
    reason: str                                  # contact-change | motion-still
    grasped: str | None
    object_poses: dict[str, Pose] = field(default_factory=dict)
    gripper: Pose | None = None


def segment_keyframes(trace: DemoTrace, still_thresh: float = 0.004) -> list[Keyframe]:
    """Keyframes occur where the demo's *meaning* changes: a contact transition
    (grasp/release) or a motion boundary (the manipulated object comes to rest).
    These become the nodes of the task graph."""
    kfs: list[Keyframe] = []
    prev_contact = {n: False for n in trace.contact}

    def push(t, reason):
        grasped = next((n for n, c in trace.contact.items() if c[t]), None)
        kfs.append(Keyframe(t=t, reason=reason, grasped=grasped,
                            object_poses={n: trace.object_tracks[n][t] for n in trace.object_tracks},
                            gripper=trace.gripper[t]))

    push(0, "start")
    for t in range(1, trace.T):
        # (1) contact transition on any object -- a grasp or release. These are
        #     the primary meaning boundaries for manipulation.
        changed = any(trace.contact[n][t] != prev_contact[n] for n in trace.contact)
        held = any(trace.contact[n][t] for n in trace.contact)
        prev_contact = {n: trace.contact[n][t] for n in trace.contact}
        if changed:
            push(t, "contact-change")
            continue
        # (2) contact-FREE motion boundary: an object being pushed/wiped (no grasp)
        #     comes to rest. Ignored while something is grasped, since a held
        #     object's micro-jitter is not a task event -- this is what kept
        #     pick-and-place graphs clean.
        if not held:
            moving = _max_object_speed(trace, t)
            moving_prev = _max_object_speed(trace, t - 1)
            if moving_prev > still_thresh and moving <= still_thresh:
                push(t, "motion-still")
    if kfs[-1].t != trace.T - 1:
        push(trace.T - 1, "end")
    return _dedup(kfs)


def _max_object_speed(trace: DemoTrace, t: int) -> float:
    if t <= 0:
        return 0.0
    return max(float(np.linalg.norm(trace.object_tracks[n][t][:3] - trace.object_tracks[n][t - 1][:3]))
               for n in trace.object_tracks)


def _dedup(kfs: list[Keyframe]) -> list[Keyframe]:
    out: list[Keyframe] = []
    for k in kfs:
        if out and out[-1].t == k.t:
            continue
        out.append(k)
    return out
