"""One recoverable disturbance per episode, injected mid-execution.

A disturbance is a callback installed on the backend that fires once, at a chosen
step, and perturbs the world in a way the system is expected to *recover* from
(not an unrecoverable catastrophe). This is what separates first-attempt success
from eventual-success-after-recovery in the metrics.
"""
from __future__ import annotations

import numpy as np

from .base import SimState, StepInfo


class Disturbance:
    def __init__(self, kind: str, target: str, at_step: int, magnitude: float,
                 rng: np.random.Generator):
        self.kind = kind          # nudge | drop | displace
        self.target = target
        self.at_step = at_step
        self.magnitude = magnitude
        self.rng = rng
        self.fired = False
        # True only if firing ACTUALLY changed the world for the target (e.g. a
        # "drop" is a no-op when the target isn't held). This is the honest
        # signal for "did the disturbance create a real recovery opportunity",
        # rather than crediting recovery whenever the episode eventually succeeds.
        self.perturbed = False

    def __call__(self, s: SimState, info: StepInfo) -> None:
        if self.fired or s.t < self.at_step or self.target in s.fallen:
            return
        self.fired = True
        obj = s.objects.get(self.target)
        if obj is None:
            return
        if self.kind == "drop" and s.grasped == self.target:
            # knock the held object out of the gripper.
            s.grasped = None
            s.gripper_closed = 0.0
            self.perturbed = True
            info.events.append(f"disturbance:drop:{self.target}")
        elif self.kind == "displace":
            direction = self.rng.normal(0, 1, size=2)
            direction /= (np.linalg.norm(direction) + 1e-9)
            obj.pose[:2] += direction * self.magnitude
            if s.grasped == self.target:
                s.grasped = None
            self.perturbed = True
            info.events.append(f"disturbance:displace:{self.target}")
        elif self.kind == "nudge":  # small planar shove
            obj.pose[:2] += self.rng.normal(0, self.magnitude, size=2)
            self.perturbed = True
            info.events.append(f"disturbance:nudge:{self.target}")


def sample_disturbance(objects: list[str], horizon: int, seed: int) -> Disturbance:
    rng = np.random.default_rng(seed + 7919)
    kind = rng.choice(["nudge", "displace", "drop"])
    target = rng.choice(objects)
    at_step = int(rng.integers(max(1, horizon // 4), max(2, 3 * horizon // 4)))
    magnitude = float(rng.uniform(0.03, 0.07))
    return Disturbance(kind, target, at_step, magnitude, rng)
