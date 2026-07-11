"""Separate development and held-out seed generators.

Thresholds and design choices must be tuned on DEV seeds only; the HELD-OUT block
is disjoint and reserved for the final reported numbers, so nothing is tuned on
the evaluation set.
"""
from __future__ import annotations

DEV_BASE = 0
HELDOUT_BASE = 1_000_000


def dev_seeds(n: int, heldout: bool = False, group: int = 0):
    base = HELDOUT_BASE if heldout else DEV_BASE
    start = base + group * 10_000
    return range(start, start + n)


def heldout_seeds(n: int, group: int = 0):
    return dev_seeds(n, heldout=True, group=group)
