"""AgentEnv: the wall between the agent and ground truth.

The agent is handed an AgentEnv, never the raw backend. AgentEnv exposes exactly
two things: `reset_percept()` and `step(action) -> Percept`. It does NOT expose
`state()`; touching that attribute raises PrivilegedAccessError. Ground-truth
StepInfo (collisions, force, irreversibility) is recorded internally for the
scorer and never returned to the agent.

The benchmark harness holds the real backend separately and reads ground truth
only through the scorer, after the episode.
"""
from __future__ import annotations

from ..perception.detections import Corruptor, CorruptionSpec, Percept
from ..sim.base import Action


class PrivilegedAccessError(RuntimeError):
    pass


class AgentEnv:
    def __init__(self, backend, corruption: CorruptionSpec | None = None,
                 rng=None):
        import numpy as np
        self._backend = backend
        self._rng = rng or np.random.default_rng(0)
        self._corruptor = Corruptor(corruption or CorruptionSpec(), self._rng)
        # ground-truth safety log, for the scorer only
        self.step_info_log: list = []
        self.n_steps = 0
        self._observation_t = 0

    def reset_percept(self) -> Percept:
        p = self._corruptor(self._backend.perceive())
        self._observation_t = p.t
        return p

    def step(self, action: Action) -> Percept:
        _, info = self._backend.step(action)
        self.step_info_log.append(info)
        self.n_steps += 1
        p = self._corruptor(self._backend.perceive())
        self._observation_t = max(self._observation_t + 1, p.t)
        p.t = self._observation_t
        return p

    def set_viewpoint(self, name: str) -> Percept:
        """Move the camera and return a fresh percept from the new view. This is an
        ACTION (it changes what is observable), NOT privileged state access."""
        if hasattr(self._backend, "set_camera"):
            self._backend.set_camera(name)
        self.n_steps += 1
        p = self._corruptor(self._backend.perceive())
        # A camera move creates a fresh observation even when the rigid-body
        # simulator's world clock has not advanced.  Give it a monotonic sensor
        # timestamp so the estimator neither discards it as stale nor double
        # counts it as the prior frame.
        self._observation_t = max(self._observation_t + 1, p.t)
        p.t = self._observation_t
        return p

    # -- guard rails ------------------------------------------------------
    def state(self, *a, **k):
        raise PrivilegedAccessError(
            "agent code may not read ground-truth SimState; use the BeliefState "
            "produced by StateEstimator from percepts")

    def __getattr__(self, name):
        # any attempt to reach through to backend internals is blocked
        if name in ("_backend", "_corruptor", "_rng"):
            raise AttributeError(name)
        raise PrivilegedAccessError(
            f"AgentEnv exposes no '{name}'; the agent sees only percepts")
