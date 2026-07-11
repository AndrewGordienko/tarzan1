"""The agent side of the wall.

Everything in this package operates ONLY on agent-visible information: noisy
Observations and the BeliefState estimated from them. Nothing here may read the
simulator's ground-truth SimState -- that is enforced by AgentEnv (which exposes
no `state()`) and by an architectural test.
"""
from .belief import BeliefObject, BeliefState
from .estimator import StateEstimator
from .env import AgentEnv, PrivilegedAccessError
from .dynamics_context import DynamicsContext

__all__ = ["BeliefObject", "BeliefState", "StateEstimator", "AgentEnv",
           "PrivilegedAccessError", "DynamicsContext"]
