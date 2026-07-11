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


@dataclass
class SkillResult:
    success: bool
    observation: Any = None
    contact_events: tuple[dict[str, Any], ...] = ()
    execution_steps: int = 0
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
