"""Agent-facing perception types + configurable corruptions.

A Percept is what a sensor produces: an UNORDERED list of nameless Detections
plus proprioception. There are deliberately no semantic names and no stable
ordering, so the agent cannot cheat by dict key or index -- it must associate
detections into tracks over time (the estimator's job) and figure out which
track plays which role (correspondence's job).

Corruptor injects the structured, temporal corruptions the benchmark requires:
extra position noise, occlusion, whole-frame drops, sensing delay, false contact
readings, and detection-identity swaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..geometry import Pose, pose


@dataclass
class Detection:
    pose: Pose
    size: np.ndarray
    shape: str = "box"
    color: str = "unknown"
    marker: str = "unknown"       # side-visible label/marker, if the view sees it
    contact: bool = False           # noisy hint that the gripper touches this
    size_meas_std: np.ndarray = None  # per-axis size measurement std (camera calibration);
                                      # large on an axis the current view cannot resolve

    def copy(self) -> "Detection":
        c = replace(self, pose=self.pose.copy(), size=self.size.copy())
        if self.size_meas_std is not None:
            c.size_meas_std = self.size_meas_std.copy()
        return c


@dataclass
class Percept:
    detections: list[Detection]
    gripper: Pose                   # proprioception (near-exact)
    gripper_closed: float
    t: int

    def copy(self) -> "Percept":
        return Percept([d.copy() for d in self.detections], self.gripper.copy(),
                       self.gripper_closed, self.t)


@dataclass
class CorruptionSpec:
    pos_noise: float = 0.004        # extra per-detection position std (m)
    size_noise: float = 0.0         # per-detection size std (m); >0 makes passive
                                    # inspection informative (independent per frame)
    occlusion_prob: float = 0.05    # per-detection chance of being missed a frame
    drop_prob: float = 0.02         # whole-frame drop (sensor returns nothing new)
    delay_frames: int = 0           # constant sensing latency, in frames
    false_contact_prob: float = 0.03
    identity_swap_prob: float = 0.02  # swap two nearby detections' apparent pose
    enabled: bool = True


class Corruptor:
    """Stateful: buffers frames for delay/drop, so temporal effects are real."""
    def __init__(self, spec: CorruptionSpec, rng: np.random.Generator):
        self.spec = spec
        self.rng = rng
        self._buffer: list[Percept] = []
        self._last_emitted: Percept | None = None

    def __call__(self, clean: Percept) -> Percept:
        s = self.spec
        if not s.enabled:
            return clean
        p = clean.copy()

        # whole-frame drop: re-emit the previous percept (stale)
        if self._last_emitted is not None and self.rng.random() < s.drop_prob:
            stale = self._last_emitted.copy()
            stale.t = clean.t
            return stale

        # per-detection occlusion + position noise
        kept = []
        for d in p.detections:
            if self.rng.random() < s.occlusion_prob:
                continue
            d.pose[:2] += self.rng.normal(0, s.pos_noise, size=2)
            if s.size_noise > 0:                 # independent per-frame size error
                d.size = d.size + self.rng.normal(0, s.size_noise, size=d.size.shape)
                # Preserve a calibrated innovation gate: the effective camera
                # variance is optical measurement variance plus this injected
                # independent sensor noise.  Leaving the old calibration here
                # made legitimate noisy observations look like identity swaps.
                base = (d.size_meas_std if d.size_meas_std is not None
                        else np.full(d.size.shape, 0.010))
                d.size_meas_std = np.sqrt(np.asarray(base, dtype=float) ** 2 + s.size_noise ** 2)
            if self.rng.random() < s.false_contact_prob:
                d.contact = not d.contact
            kept.append(d)
        p.detections = kept

        # identity swap: exchange the apparent poses of two nearby detections
        if len(p.detections) >= 2 and self.rng.random() < s.identity_swap_prob:
            i, j = self.rng.choice(len(p.detections), size=2, replace=False)
            p.detections[i].pose, p.detections[j].pose = (
                p.detections[j].pose, p.detections[i].pose)

        # shuffle so order carries no information
        self.rng.shuffle(p.detections)

        # constant sensing delay via a ring buffer
        if s.delay_frames > 0:
            self._buffer.append(p)
            if len(self._buffer) > s.delay_frames:
                out = self._buffer.pop(0)
            else:
                out = Percept([], p.gripper, p.gripper_closed, p.t)  # nothing yet
        else:
            out = p
        self._last_emitted = out
        return out
