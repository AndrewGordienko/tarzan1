"""DynamicsContext: online estimation of hidden physics from interaction history.

RMA-style: infer latent dynamics from recent (action, observed-effect) pairs, no
weight updates. We estimate four quantities and, honestly, they differ in how
observable they are in a kinematic tabletop sim:

  * actuator_delay  -- well observed (TCP move / commanded move each step).
  * grasp_stability -- well observed (does the inferred grasp persist while the
                       gripper is commanded closed).
  * friction_scale  -- partially observed (only revealed by post-contact slide;
                       stays near prior with high variance if no slip is seen).
  * mass_scale      -- poorly observed here (a purely kinematic carry reveals
                       little); we keep the prior and report the resulting error
                       rather than pretend to infer it.

`error_vs` compares estimates to ground truth so context-estimation error is
reported SEPARATELY from task success, as requested.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .belief import BeliefState


@dataclass
class DynamicsContext:
    actuator_delay: float = 0.28
    friction_scale: float = 1.0
    mass_scale: float = 1.0
    grasp_stability: float = 1.0
    delay_std: float = 0.12
    friction_std: float = 0.3
    mass_std: float = 0.5
    # bookkeeping
    _closed_steps: int = 0
    _grasp_steps: int = 0
    _delay_samples: list = field(default_factory=list)

    def update(self, prev: BeliefState, action_target, cur: BeliefState) -> None:
        cmd = np.asarray(action_target)[:3] - prev.gripper[:3]
        moved = cur.gripper[:3] - prev.gripper[:3]
        cmd_n = float(np.linalg.norm(cmd))
        if cmd_n > 1e-4:
            realized = float(np.clip(np.linalg.norm(moved) / cmd_n, 0.0, 1.0))
            est = float(np.clip(1.0 - realized, 0.0, 0.95))
            self._delay_samples.append(est)
            self.actuator_delay = 0.6 * self.actuator_delay + 0.4 * est
            self.delay_std = max(0.02, self.delay_std * 0.85)

        # grasp stability: of the steps commanded closed, how often is a grasp held
        if prev.gripper_closed >= 0.6:
            self._closed_steps += 1
            if cur.grasped is not None:
                self._grasp_steps += 1
            if self._closed_steps >= 3:
                self.grasp_stability = self._grasp_steps / self._closed_steps

        # friction: only observable via post-release slide of a just-placed object
        if prev.grasped is not None and cur.grasped is None and prev.grasped in cur.objects:
            slide = float(np.linalg.norm(cur.objects[prev.grasped].pose[:2]
                                         - prev.objects[prev.grasped].pose[:2]))
            if slide > 1e-3:
                # more slide -> lower friction estimate
                self.friction_scale = float(np.clip(1.0 - 2.0 * slide, 0.3, 1.5))
                self.friction_std = max(0.05, self.friction_std * 0.7)
            else:
                self.friction_std = max(0.1, self.friction_std * 0.95)

    def error_vs(self, true_delay, true_friction, true_mass) -> dict:
        return {
            "actuator_delay": abs(self.actuator_delay - true_delay),
            "friction_scale": abs(self.friction_scale - true_friction),
            "mass_scale": abs(self.mass_scale - true_mass),
        }

    def sample_params(self, rng: np.random.Generator, n: int):
        for _ in range(n):
            yield (
                float(np.clip(rng.normal(self.actuator_delay, self.delay_std), 0.0, 0.9)),
                float(np.clip(rng.normal(self.friction_scale, self.friction_std), 0.2, 2.0)),
                float(np.clip(rng.normal(self.mass_scale, self.mass_std), 0.3, 2.5)),
            )
