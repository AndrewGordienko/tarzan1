"""Minimal SE(2)+z geometry used across the whole pipeline.

We represent a pose as a 4-vector ``[x, y, z, yaw]``. This is deliberately
lighter than full SE(3): the tabletop tasks in the benchmark only need planar
translation, a lift axis, and a wrist yaw. The abstraction (relative transforms,
compose/invert) is the same one a real SE(3) system would use, so the compiler
and skill code below are unchanged when a full-6-DoF backend replaces the toy.
"""
from __future__ import annotations

import numpy as np

Pose = np.ndarray  # shape (4,): x, y, z, yaw


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> Pose:
    return np.array([x, y, z, yaw], dtype=float)


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative(a: Pose, b: Pose) -> Pose:
    """Pose of ``b`` expressed in the frame of ``a`` (i.e. a^-1 . b).

    Translation is rotated into a's yaw frame so the transform is invariant to
    where the reference object sits or how it is turned -- this invariance is
    what lets one demonstration transfer to a randomized scene.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = np.cos(-a[3]), np.sin(-a[3])
    return pose(c * dx - s * dy, s * dx + c * dy, b[2] - a[2], wrap_angle(b[3] - a[3]))


def apply(a: Pose, rel: Pose) -> Pose:
    """Compose: return the world pose of ``rel`` given reference frame ``a``."""
    c, s = np.cos(a[3]), np.sin(a[3])
    return pose(a[0] + c * rel[0] - s * rel[1],
                a[1] + s * rel[0] + c * rel[1],
                a[2] + rel[2], wrap_angle(a[3] + rel[3]))


def dist_xy(a: Pose, b: Pose) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def dist_xyz(a: Pose, b: Pose) -> float:
    return float(np.linalg.norm(a[:3] - b[:3]))
