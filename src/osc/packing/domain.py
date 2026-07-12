"""Packing domain state and the compiled, reusable task program."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from itertools import permutations
import json
from pathlib import Path
import copy
import numpy as np


@dataclass
class Container:
    container_id: str
    dimensions: tuple[float, float, float]
    max_load: float = 100.0

    @property
    def volume(self):
        return float(np.prod(self.dimensions))


@dataclass
class PackItem:
    item_id: str
    dimensions: tuple[float, float, float]
    mass: float = 1.0
    mass_uncertainty: float = 0.0
    fragile: bool = False
    load_limit: float = 100.0
    allowed_orientations: tuple[tuple[int, int, int], ...] | None = None
    metadata: dict = field(default_factory=dict)
    already_packed: bool = False

    def orientations(self):
        if self.allowed_orientations is not None:
            return [tuple(self.dimensions[i] for i in p) for p in self.allowed_orientations]
        return sorted(set(tuple(self.dimensions[i] for i in p) for p in permutations(range(3))))

    @property
    def volume(self):
        return float(np.prod(self.dimensions))


@dataclass
class Placement:
    item_id: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    orientation: tuple[int, int, int] = (0, 1, 2)
    support_id: str | None = None
    staged: bool = False

    @property
    def max_corner(self):
        return tuple(a + b for a, b in zip(self.position, self.size))

    @property
    def volume(self):
        return float(np.prod(self.size))


@dataclass
class PackingConstraint:
    name: str
    kind: str
    hard: bool = True
    weight: float = 1.0
    parameters: dict = field(default_factory=dict)


@dataclass
class PackingState:
    container: Container
    items: dict[str, PackItem]
    placements: dict[str, Placement] = field(default_factory=dict)
    staged: dict[str, Placement] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def clone(self):
        return PackingState(self.container, copy.deepcopy(self.items),
                             copy.deepcopy(self.placements), copy.deepcopy(self.staged),
                             list(self.history))

    @property
    def packed_volume(self):
        return sum(p.volume for p in self.placements.values())

    @property
    def packed_mass(self):
        return sum(self.items[i].mass for i in self.placements)

    def occupancy(self, exclude: str | None = None):
        return [p for i, p in self.placements.items() if i != exclude]


@dataclass
class PackingProgram:
    objective: list[str]
    hard_constraints: list[PackingConstraint]
    preferences: list[PackingConstraint]
    available_actions: list[str]
    ordering_rules: list[str] = field(default_factory=list)
    demonstration_events: list[dict] = field(default_factory=list)
    hidden_information: list[str] = field(default_factory=list)
    source: str = "one_shot_demo"
    policy_name: str = "heavy_bottom_fragile_top"
    posterior: dict = field(default_factory=dict)
    constraint_posterior: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
