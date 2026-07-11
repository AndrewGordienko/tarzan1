from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .commands import SkillCommand


@dataclass
class ObservationFrame:
    """Camera and robot signals exposed to Tarzan; no simulator state is included."""

    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    gripper: float | None = None
    contacts: tuple[dict[str, Any], ...] = ()
    timestamp: float = 0.0


class TinyVLAMuJoCoAdapter:
    """Optional adapter for TinyVLA's SO101Env.

    The import is deliberately lazy: core Tarzan and oracle tests do not require
    MuJoCo. Deployed observations must come through ``observe`` rather than a
    ground-truth simulator accessor.
    """

    def __init__(self, env: Any | None = None, **env_kwargs: Any):
        self.env = env
        self.env_kwargs = env_kwargs
        self._time = 0.0

    def reset(self) -> ObservationFrame:
        if self.env is None:
            try:
                from tinyvla.env import SO101Env
            except ImportError as exc:
                raise RuntimeError("Install TinyVLA/MuJoCo to run the embodied lane") from exc
            self.env = SO101Env(**self.env_kwargs)
        self.env.reset()
        return self.observe()

    def observe(self) -> ObservationFrame:
        if self.env is None:
            raise RuntimeError("adapter is not initialized; call reset() first")
        obs = getattr(self.env, "observation", None)
        if callable(obs):
            obs = obs()
        if obs is None:
            obs = getattr(self.env, "_last_obs", {})
        if not isinstance(obs, dict):
            obs = {"rgb": obs}
        return ObservationFrame(rgb=obs.get("rgb"), depth=obs.get("depth"),
                               masks=obs.get("masks", {}), gripper=obs.get("gripper"),
                               contacts=tuple(obs.get("contacts", ())), timestamp=self._time)

    def dispatch(self, command: SkillCommand) -> ObservationFrame:
        """Dispatch an intent command; continuous execution remains controller-owned."""
        if self.env is None:
            raise RuntimeError("adapter is not initialized; call reset() first")
        if command.kind not in {"inspect", "pick", "place", "temporarily_remove", "verify", "repack"}:
            raise ValueError(f"unsupported skill command: {command.kind}")
        step = getattr(self.env, "step_skill", None)
        if step is None:
            raise RuntimeError("TinyVLA adapter requires an environment skill bridge")
        step(command)
        self._time += 1.0
        return self.observe()

