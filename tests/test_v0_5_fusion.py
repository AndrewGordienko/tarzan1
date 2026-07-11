"""v0.5 Step-5 (passive multi-frame inspection): additional independent frames
must genuinely reduce estimator covariance AND size RMSE, with diminishing
returns and NO reduction from stale/duplicate frames.
"""
import numpy as np

from osc.agent.estimator import StateEstimator, SIZE_MEAS_VAR
from osc.perception.detections import Detection, Percept


def _percept(size, t, noise, rng):
    d = Detection(pose=np.array([0.4, 0.0, 0.02, 0.0]),
                  size=size + rng.normal(0, noise, size=3), shape="box", color="red")
    return Percept(detections=[d], gripper=np.zeros(4), gripper_closed=0.0, t=t)


def test_size_covariance_and_rmse_fall_with_independent_frames():
    rng = np.random.default_rng(0)
    true = np.array([0.045, 0.045, 0.045])
    est = StateEstimator()
    stds, rmses = [], []
    for t in range(1, 21):
        b = est.update(_percept(true, t, 0.010, rng))
        o = next(iter(b.objects.values()))
        stds.append(o.size_std)
        rmses.append(float(np.sqrt(np.mean((o.size - true) ** 2))))
    # covariance falls monotonically and roughly as 1/sqrt(N)
    assert stds[0] > stds[4] > stds[19]
    assert stds[19] < 0.4 * stds[0]
    # posterior size std tracks the Kalman prediction ~ sqrt(MEAS/N)
    expected = np.sqrt(SIZE_MEAS_VAR / 20)
    assert 0.5 * expected < stds[19] < 2.0 * expected
    # the estimate actually converges toward truth
    assert rmses[19] < rmses[0]
    assert rmses[19] < 0.006


def test_stale_duplicate_frame_does_not_shrink_covariance():
    rng = np.random.default_rng(1)
    true = np.array([0.045, 0.045, 0.045])
    est = StateEstimator()
    est.update(_percept(true, 1, 0.010, rng))
    est.update(_percept(true, 2, 0.010, rng))
    o = next(iter(est.tracks.values()))
    before = o.size_std
    # re-emit the SAME frame index (t unchanged) -> must not fuse again
    est.update(_percept(true, 2, 0.010, rng))
    assert next(iter(est.tracks.values())).size_std == before


def test_zero_size_noise_leaves_default_benchmark_untouched():
    # with no size noise configured, sizes are exact and fusion just confirms them.
    rng = np.random.default_rng(2)
    true = np.array([0.05, 0.05, 0.05])
    est = StateEstimator()
    for t in range(1, 6):
        b = est.update(_percept(true, t, 0.0, rng))
    o = next(iter(b.objects.values()))
    assert np.allclose(o.size, true, atol=1e-6)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
