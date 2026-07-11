from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkillCommand:
    """Intent-level command exchanged with a continuous controller."""

    kind: str
    object_query: dict[str, Any] = field(default_factory=dict)
    target_region: dict[str, Any] = field(default_factory=dict)
    orientation: tuple[float, ...] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

