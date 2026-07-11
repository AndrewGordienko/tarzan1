from __future__ import annotations

import numpy as np

from .mujoco_adapter import ObservationFrame


def estimate_from_segmentation(frame: ObservationFrame, intrinsics: dict[str, float] | None = None) -> dict:
    """Estimate object extents from masks and depth (the first embodied ladder rung)."""
    if frame.depth is None:
        raise ValueError("segmentation+depth perception requires a depth image")
    intrinsics = intrinsics or {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    result = {}
    for name, mask in frame.masks.items():
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        z = np.asarray(frame.depth)[ys, xs]
        z = z[np.isfinite(z) & (z > 0)]
        if len(z) == 0:
            continue
        result[name] = {"center": (float(xs.mean()), float(ys.mean()), float(z.mean())),
                        "size": (float(np.ptp(xs)), float(np.ptp(ys)), float(np.std(z) * 2)),
                        "covariance": (0.01, 0.01, max(1e-4, float(np.var(z))))}
    return result
