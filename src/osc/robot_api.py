"""Hardware-independent skill boundary used by Tarzan embodiments."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class ActuatorCommand:
    joint_targets: tuple[float, ...]
    gripper_targets: tuple[float, ...]
    phase: str

@dataclass(frozen=True)
class RobotObservation:
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    gripper_position: tuple[float, ...]
    contact_forces_n: tuple[float, ...]
    object_estimates: Mapping[str, object]

class RobotArmAPI(ABC):
    @abstractmethod
    def reset(self, seed: int = 0) -> RobotObservation: ...
    @abstractmethod
    def observe(self) -> RobotObservation: ...
    @abstractmethod
    def step(self, command: ActuatorCommand) -> RobotObservation: ...
    @abstractmethod
    def verify(self) -> Mapping[str, object]: ...
