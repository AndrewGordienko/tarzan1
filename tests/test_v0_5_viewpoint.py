"""v0.5 Step-5 (viewpoint evidence): a size axis occluded from the default camera
is revealed only by moving to the correct viewpoint -- through the normal sensor
+ estimator path, with per-axis calibrated fusion. No 'reveal correct role' field.
"""
import numpy as np

from osc.agent.env import AgentEnv, PrivilegedAccessError
from osc.agent.estimator import StateEstimator
from osc.perception.detections import CorruptionSpec
from osc.sim.base import SimObject, SimState
from osc.sim.toy import ToyTabletopSim


def _state():
    # one object; true HEIGHT (z) 0.06, footprint 0.045 -- height is the feature
    o = SimObject(name="A", size=np.array([0.045, 0.045, 0.060]),
                  pose=np.array([0.4, 0.0, 0.03, 0.0]))
    return SimState(objects={"A": o}, gripper=np.zeros(4), gripper_closed=False,
                    grasped=None, table_bounds=(0, 1, -0.5, 0.5), table_z=0.0, fallen=set(), t=0)


def _z_error_after(camera, frames, seed=0):
    sim = ToyTabletopSim(camera_model=True, rng=np.random.default_rng(seed))
    sim._s = _state(); sim.set_camera(camera)
    est = StateEstimator()
    b = None
    for t in range(1, frames + 1):
        p = sim.perceive(); p.t = t
        b = est.update(p)
    o = next(iter(b.objects.values()))
    return abs(float(o.size[2]) - 0.060)          # height-estimate error


def test_default_top_view_cannot_resolve_height():
    # many frames from the default top view do NOT pin the height down
    assert _z_error_after("top", 20) > 0.015


def test_correct_viewpoint_reveals_height():
    # the front view sees height; its estimate converges
    assert _z_error_after("front", 20) < 0.006


def test_wrong_alternative_viewpoint_does_not_help():
    # a different view that still looks down the height axis reveals nothing
    assert _z_error_after("top_rot", 20) > 0.015


def test_side_view_also_reveals_height_but_top_does_not():
    assert _z_error_after("side", 20) < 0.006
    assert _z_error_after("top", 20) > _z_error_after("front", 20)


def test_detection_carries_per_axis_calibration():
    sim = ToyTabletopSim(camera_model=True, rng=np.random.default_rng(0))
    sim._s = _state(); sim.set_camera("top")
    d = sim.perceive().detections[0]
    assert d.size_meas_std is not None
    assert d.size_meas_std[2] > d.size_meas_std[0]   # height blurred, footprint sharp


def test_set_viewpoint_is_an_action_not_privileged_state():
    sim = ToyTabletopSim(camera_model=True); sim._s = _state()
    env = AgentEnv(sim, CorruptionSpec(enabled=False), rng=np.random.default_rng(0))
    p = env.set_viewpoint("front")                    # allowed: it's an action
    assert p.detections
    assert sim.camera == "front"
    try:
        env.state(); assert False
    except PrivilegedAccessError:
        pass


def test_camera_model_off_leaves_sizes_exact():
    sim = ToyTabletopSim(camera_model=False); sim._s = _state()
    d = sim.perceive().detections[0]
    assert np.allclose(d.size, [0.045, 0.045, 0.060], atol=1e-9)
    assert d.size_meas_std is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
