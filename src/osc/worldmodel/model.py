"""Action-conditioned world model used for imagined search and adaptation.

For the toy backend this is a *parameter-ensemble* forward model: the system does
NOT know the true friction / mass / actuator-delay of the current episode, so the
world model carries a small ensemble of plausible dynamics and rolls a candidate
plan through all of them. Ensemble agreement gives success probability; ensemble
disagreement gives calibrated uncertainty. Collision / force / irreversibility
are read from the simulated StepInfo. In the full system this ensemble is
replaced by a learned latent forward model (V-JEPA-2 / OSVI-WM style) exposing
the same `rollout` interface.

`update_context` implements RMA-style online adaptation: after each real step we
nudge the belief toward parameters that best explain the observed motion, without
any gradient update to a policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sim.base import SimState
from ..sim.toy import ToyTabletopSim


@dataclass
class RolloutResult:
    final_state: SimState
    success_prob: float
    collision_risk: float
    force_risk: float
    irreversible_risk: float
    uncertainty: float
    steps: int
    traces: list = field(default_factory=list)   # per-ensemble final object poses


class WorldModel:
    def __init__(self, ensemble_size: int = 5, seed: int = 0, max_steps: int = 40):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        # belief over dynamics: (actuator_delay, friction_scale, mass_scale)
        self.belief = {
            "actuator_delay": 0.28,
            "friction_scale": 1.0,
            "mass_scale": 1.0,
            "delay_std": 0.12, "friction_std": 0.25, "mass_std": 0.4,
        }
        self.ensemble_size = ensemble_size

    # -- online adaptation (RMA-style, no gradient) -----------------------
    def update_context(self, prev_state: SimState, action_target, next_state: SimState) -> None:
        """Estimate effective actuator delay from how far the TCP actually moved
        toward its commanded target, and shrink belief uncertainty accordingly."""
        cmd = np.asarray(action_target)[:3] - prev_state.gripper[:3]
        moved = next_state.gripper[:3] - prev_state.gripper[:3]
        cmd_n = np.linalg.norm(cmd)
        if cmd_n > 1e-4:
            realized = float(np.clip(np.linalg.norm(moved) / cmd_n, 0.0, 1.0))
            est_delay = float(np.clip(1.0 - realized, 0.0, 0.95))
            # exponential moving update of the mean, with variance decay
            self.belief["actuator_delay"] = 0.6 * self.belief["actuator_delay"] + 0.4 * est_delay
            self.belief["delay_std"] = max(0.02, self.belief["delay_std"] * 0.8)

    # -- imagined rollout -------------------------------------------------
    def _sample_params(self):
        b = self.belief
        for _ in range(self.ensemble_size):
            yield (
                float(np.clip(self.rng.normal(b["actuator_delay"], b["delay_std"]), 0.0, 0.9)),
                float(np.clip(self.rng.normal(b["friction_scale"], b["friction_std"]), 0.2, 2.0)),
                float(np.clip(self.rng.normal(b["mass_scale"], b["mass_std"]), 0.3, 2.5)),
            )

    def rollout(self, state: SimState, plan, goal_check) -> RolloutResult:
        """Roll `plan` (a list of SkillInstance) through each ensemble member.
        `goal_check(state) -> bool` decides success."""
        successes, collisions, forces, irreversibles, steps_used = [], [], [], [], []
        finals = []
        for delay, fr_scale, mass_scale in self._sample_params():
            sim = ToyTabletopSim(actuator_delay=delay, lighting=1.0,
                                 rng=np.random.default_rng(self.rng.integers(1 << 30)))
            s = state.copy()
            for o in s.objects.values():
                o.friction *= fr_scale
                o.mass *= mass_scale
            sim.reset(s)
            col = frc = irr = 0
            n = 0
            for si in plan:
                for _ in range(self.max_steps):
                    if si.done(sim.state()):
                        break
                    action = si.act(sim.state())
                    _, info = sim.step(action)
                    n += 1
                    col += int(info.collision)
                    frc += int(info.force_violation)
                    irr += int(info.irreversible)
                    if n >= self.max_steps * max(1, len(plan)):
                        break
            fs = sim.state()
            finals.append({k: v.pose.copy() for k, v in fs.objects.items()})
            successes.append(float(goal_check(fs)))
            collisions.append(col); forces.append(frc); irreversibles.append(irr)
            steps_used.append(n)

        # uncertainty = mean pairwise spread of final object positions across ensemble
        uncertainty = _ensemble_spread(finals)
        return RolloutResult(
            final_state=state,  # representative; caller uses risks, not this
            success_prob=float(np.mean(successes)),
            collision_risk=float(np.mean(collisions)),
            force_risk=float(np.mean(forces)),
            irreversible_risk=float(np.mean(irreversibles)),
            uncertainty=uncertainty,
            steps=int(np.mean(steps_used)),
            traces=finals)


def _ensemble_spread(finals: list[dict]) -> float:
    if len(finals) < 2:
        return 0.0
    names = finals[0].keys()
    spreads = []
    for n in names:
        pts = np.stack([f[n][:3] for f in finals])
        spreads.append(float(np.mean(np.std(pts, axis=0))))
    return float(np.mean(spreads))
