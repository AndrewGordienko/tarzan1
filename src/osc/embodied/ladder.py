from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LadderConfig:
    perception: str
    acquisition: str
    execution: str


LADDER = (
    LadderConfig("oracle", "oracle_program", "oracle_arm"),
    LadderConfig("segdepth", "structured_events", "scripted"),
    LadderConfig("rgbd", "camera_events", "scripted"),
    LadderConfig("rgb", "camera_events", "tinyvla"),
)


def unavailable_report(episodes: int, config: LadderConfig, error: str) -> dict:
    return {"status": "blocked", "episodes": episodes, "config": asdict(config), "error": error,
            "ground_truth_used": False}

