"""Packing demonstration events."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PackingEvent:
    kind: str
    item_id: str | None = None
    container_id: str | None = None
    support_id: str | None = None
    properties: dict | None = None

    def to_dict(self):
        return asdict(self)


def parse_demo(events):
    """Normalize scripted/observed event records into packing events."""
    out = []
    for e in events:
        if isinstance(e, PackingEvent):
            out.append(e)
        else:
            out.append(PackingEvent(**e))
    return out


def canonical_demo_events():
    return parse_demo([
        PackingEvent("inspect", properties={"container": "box_demo"}),
        PackingEvent("pick", "heavy"), PackingEvent("place_inside", "heavy", "box_demo"),
        PackingEvent("pick", "long"), PackingEvent("place_inside", "long", "box_demo"),
        PackingEvent("pick", "fragile"), PackingEvent("place_inside", "fragile", "box_demo"),
        PackingEvent("pick", "ordinary"), PackingEvent("place_inside", "ordinary", "box_demo"),
        PackingEvent("verify", container_id="box_demo"),
    ])
