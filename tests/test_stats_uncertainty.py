import numpy as np

import stats
import uncertainty as U


def test_window_sq_err_shape_and_value():
    true = np.zeros((10, 3, 30))
    pred = np.full((10, 3, 30), 2.0)
    e = stats.window_sq_err(pred, true, 7)
    assert e.shape == (10,) and np.allclose(e, 4.0)


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(0)
    e_good = rng.uniform(0.5, 1.5, 200)
    e_bad = e_good * 4  # model A has half the RMSE of model B
    d, lo, hi, p = stats.block_bootstrap_rmse_diff(e_good, e_bad, n_boot=500)
    assert d < 0 and hi < 0 and p < 0.05


def test_bootstrap_no_difference_ci_contains_zero():
    rng = np.random.default_rng(1)
    e = rng.uniform(0.5, 1.5, 200)
    d, lo, hi, p = stats.block_bootstrap_rmse_diff(e, e.copy(), n_boot=500)
    assert d == 0 and lo <= 0 <= hi


def test_conformal_quantile_finite_sample_correction():
    res = np.arange(1, 101)  # 1..100
    assert U.conformal_quantile(res, 0.1) == 91  # ceil(101*0.9) = 91
    assert U.conformal_quantile(res, 0.0) == 100  # capped at the maximum


def test_static_conformal_reaches_nominal_coverage_when_exchangeable():
    rng = np.random.default_rng(0)
    val = np.abs(rng.normal(0, 1, (400, 5)))
    test_res = np.abs(rng.normal(0, 1, (400, 5)))
    pred = np.zeros((400, 5))
    lo, hi = U.intervals_static(val, pred, alpha=0.1)
    cov = ((test_res <= hi[0, 0]) & True).mean()
    assert 0.86 < cov < 0.94


def test_rolling_conformal_uses_only_observed_residuals():
    """Residual of a *future* window (not yet observed at forecast time) must not change the interval."""
    n_val, n_test, h = 10, 20, 7
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(size=(n_val + n_test, 3)))
    pred = np.zeros((n_test, 3))
    lo1, hi1 = U.intervals_rolling(base, n_val, pred, h, alpha=0.1, window=10)
    tampered = base.copy()
    tampered[n_val + 10:] += 100.0  # residuals of windows 10.. are unknown when forecasting window 10
    lo2, hi2 = U.intervals_rolling(tampered, n_val, pred, h, alpha=0.1, window=10)
    assert np.allclose(hi1[:10 + h], hi2[:10 + h])  # windows < 17 only see residuals observed before test window 10
    assert not np.allclose(hi1[-1], hi2[-1])  # but later windows may use it


def test_evaluate_coverage_and_width():
    true = np.array([[5.0, 10.0]])
    lo, hi = np.log1p(np.array([[4.0, 11.0]])), np.log1p(np.array([[6.0, 12.0]]))
    r = U.evaluate(lo, hi, true)
    assert r["coverage"] == 0.5 and np.isclose(r["width"], 1.5)


def test_rolling_conformal_needs_enough_calibration_windows():
    import pytest
    with pytest.raises(ValueError, match="calibration"):
        U.intervals_rolling(np.ones((6, 2)), 3, np.zeros((3, 2)), 7, 0.1)
