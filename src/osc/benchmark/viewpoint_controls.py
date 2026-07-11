"""Decisive oracle-track controls for genuinely non-redundant viewpoints.

These are intentionally small inference controls, not execution tasks: track
identity is held oracle-stable while the evidence exposed to correspondence is
changed from TOP -> correct/incorrect view.  They rule out object names, colour,
and detection ordering as explanations for a viewpoint result.
"""
from __future__ import annotations

import numpy as np

from ..agent.belief import BeliefObject, BeliefState
from ..geometry import pose
from ..skills.correspondence import RoleBelief


ROLES = ("manipuland0", "support0")


def _obj(tid, size, size_var, marker="unknown"):
    return BeliefObject(track_id=tid, pose=pose(0.15 if tid == "q7" else -0.15, 0.0, 0.02),
                        size=np.asarray(size, dtype=float), size_var=np.asarray(size_var, dtype=float),
                        size_std=float(np.sqrt(np.max(size_var))), shape="box", color="unknown",
                        marker=marker)


def _evaluate(sigs, sig_vars, objects, truth):
    ra = RoleBelief(sigs, sig_vars).update(BeliefState(objects=objects))
    return dict(mapping=ra.mapping, ambiguous=ra.ambiguous, confidence=ra.confidence,
                binding_correct=all(ra.mapping.get(r) == truth[r] for r in truth),
                observed_dims={tid: obj.feature_var().tolist() for tid, obj in objects.items()})


def geometry_height_control() -> dict:
    """Same top silhouette; only a side view observes the distinct heights."""
    sigs = {
        "manipuland0": np.array([.040, .030, 0., 0., .75]),
        "support0": np.array([.040, .070, 0., 0., .75]),
    }
    sig_vars = {r: np.array([4e-6, 4e-6, 1e-6, 1., 1.]) for r in ROLES}
    truth = {"manipuland0": "q7", "support0": "q2"}
    top = {"q7": _obj("q7", [.040, .040, .040], [4e-6, 4e-6, 1.0]),
           "q2": _obj("q2", [.040, .040, .040], [4e-6, 4e-6, 1.0])}
    side = {"q7": _obj("q7", [.040, .040, .030], [1.0, 4e-6, 4e-6]),
            "q2": _obj("q2", [.040, .040, .070], [1.0, 4e-6, 4e-6])}
    # top_rot is an actual wrong geometry: it leaves height unobserved.
    return dict(oracle_tracks_top_only=_evaluate(sigs, sig_vars, top, truth),
                oracle_tracks_correct_view=_evaluate(sigs, sig_vars, side, truth),
                oracle_tracks_wrong_view=_evaluate(sigs, sig_vars, top, truth))


def side_marker_control() -> dict:
    """Geometry is identical; only a marker visible from SIDE disambiguates."""
    sigs = {
        "manipuland0": np.array([.040, .040, 0., 0., 0.00]),  # alpha
        "support0": np.array([.040, .040, 0., 0., 0.25]),     # beta
    }
    sig_vars = {r: np.array([4e-6, 4e-6, 1e-6, 1., 1e-6]) for r in ROLES}
    truth = {"manipuland0": "q7", "support0": "q2"}
    top = {"q7": _obj("q7", [.040, .040, .040], [4e-6, 4e-6, 1.0]),
           "q2": _obj("q2", [.040, .040, .040], [4e-6, 4e-6, 1.0])}
    side = {"q7": _obj("q7", [.040, .040, .040], [1.0, 4e-6, 4e-6], marker="alpha"),
            "q2": _obj("q2", [.040, .040, .040], [1.0, 4e-6, 4e-6], marker="beta")}
    return dict(oracle_tracks_top_only=_evaluate(sigs, sig_vars, top, truth),
                oracle_tracks_correct_view=_evaluate(sigs, sig_vars, side, truth),
                oracle_tracks_wrong_view=_evaluate(sigs, sig_vars, top, truth))


def run_viewpoint_positive_controls() -> dict:
    return {"geometry_height": geometry_height_control(), "side_marker": side_marker_control()}
