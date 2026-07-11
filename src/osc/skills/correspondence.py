"""Bind task-graph ROLES to concrete estimator tracks at eval time.

The demo stored a feature signature per role (size, shape, coarse color). Given
the eval BeliefState -- whose tracks have anonymous IDs, randomized names,
shuffled order, and possibly extra distractor objects -- we assign each role to
the best-matching track by weighted feature distance, uniquely, leaving
distractors unmatched. This is what forces the system to pick the demonstrated
manipuland/target by geometry+appearance rather than by name or dict order.

Weights emphasize size/shape over color, since color is heavily randomized.
"""
from __future__ import annotations

import numpy as np

from ..agent.belief import BeliefState

W = np.array([3.0, 3.0, 1.0, 0.3])      # size_x, size_z, shape, color


def correspond(belief: BeliefState, role_signatures: dict) -> dict:
    tracks = list(belief.objects.values())
    roles = [r for r, s in role_signatures.items() if s is not None]
    # cost matrix role x track
    pairs = []
    for r in roles:
        sig = np.asarray(role_signatures[r], dtype=float)
        for o in tracks:
            d = float(np.linalg.norm(W * (sig - o.feature())))
            pairs.append((d, r, o.track_id))
    pairs.sort(key=lambda x: x[0])
    mapping, used_roles, used_tracks = {}, set(), set()
    for d, r, tid in pairs:
        if r in used_roles or tid in used_tracks:
            continue
        mapping[r] = tid
        used_roles.add(r); used_tracks.add(tid)
    return mapping
