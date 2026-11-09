import json
import os
import sys

import numpy as np
import pytest

import train


def test_split_is_chronological_and_leakage_free():
    T = 1104
    tr, va, te = train.make_splits(T)
    assert tr.max() < va.min() < va.max() < te.min()
    assert tr.max() + train.H < va.min(), "training targets must not reach into the validation period"
    assert va.max() + train.H < te.min(), "validation targets must not reach into the test period"
    assert te.max() + train.H <= T - 1 and tr.min() >= train.L - 1


def test_split_proportions():
    tr, va, te = train.make_splits(1104)
    n = len(tr) + len(va) + len(te) + 2 * train.H
    assert abs((len(tr) + train.H) / n - 0.70) < 0.02 and abs(len(te) / n - 0.20) < 0.02


def test_short_series_raises_clear_error():
    with pytest.raises(ValueError, match="too short"):
        train.make_splits(120)


def test_windows_and_targets_alignment():
    T, N, Fd = 100, 4, 3
    X = np.arange(T, dtype=np.float32)[:, None, None] * np.ones((1, N, Fd), dtype=np.float32)
    idx = np.array([20, 40])
    w = train.windows(X, idx)
    y = train.targets(X[..., 0], idx)
    assert w.shape == (2, train.L, N, Fd) and y.shape == (2, N, train.H)
    assert w[0, -1, 0, 0] == 20 and w[0, 0, 0, 0] == 20 - train.L + 1  # window ends at the origin
    assert y[0, 0, 0] == 21 and y[0, 0, -1] == 20 + train.H           # targets strictly in the future


def test_metrics_perfect_forecast_is_zero():
    y = np.random.default_rng(0).uniform(1, 50, (5, 3, train.H))
    m = train.metrics(y, y)
    for h in train.HORIZONS:
        assert m[h]["RMSE"] == m[h]["MAE"] == m[h]["MAPE"] == 0
    assert len(m["rmse_curve"]) == train.H


def test_metrics_known_values():
    t = np.full((2, 2, train.H), 10.0)
    p = t + 3.0
    m = train.metrics(p, t)
    assert np.isclose(m[7]["RMSE"], 3) and np.isclose(m[7]["MAE"], 3) and np.isclose(m[7]["MAPE"], 30)


def test_mape_ignores_near_zero_truth():
    t = np.zeros((1, 2, train.H)); t[0, 0, :] = 10.0
    p = t + 1.0
    m = train.metrics(p, t)
    assert np.isclose(m[7]["MAPE"], 10.0)  # only the y>=1 entries count


def test_to_cases_nonnegative_and_clipped():
    out = train.to_cases(np.array([-5.0, 0.0, np.log1p(10), 100.0]))
    assert out[0] == 0 and out[1] == 0 and np.isclose(out[2], 10) and np.isfinite(out[3])


@pytest.mark.parametrize("name", ["qepignn", "gcn_lstm"])
def test_short_training_run_learns_and_is_deterministic(name, synthetic_data):
    pred1, info, extra = train.run_neural(name, 0, synthetic_data, epochs=3, lr=3e-3, d=16)
    pred2, _, _ = train.run_neural(name, 0, synthetic_data, epochs=3, lr=3e-3, d=16)
    n_te, N = len(synthetic_data[5][2]), synthetic_data[3].shape[0]
    assert pred1.shape == (n_te, N, train.H) and np.isfinite(pred1).all()
    assert np.allclose(pred1, pred2, atol=1e-5), "same seed must give the same result"
    assert info["val_mse"] < 5 and info["params"] > 0
    if name == "qepignn":
        assert extra["influence"].shape == (N, N) and len(extra["walk_times"]) == 4


@pytest.mark.skipif(not os.path.exists("data/processed/covid_us_states.npz"), reason="processed data not present")
def test_cli_end_to_end_on_real_processed_data(tmp_path, monkeypatch):
    """Runs the real entry point (main) with baselines + one tiny neural run; writes into a temp dir."""
    monkeypatch.setattr(train, "RES", str(tmp_path) + "/")
    monkeypatch.setattr(sys, "argv", ["train.py", "--models", "persistence", "ar_ridge", "lstm",
                                      "--seeds", "0", "--epochs", "1", "--out", "r.json"])
    train.main()
    res = json.load(open(tmp_path / "r.json"))
    assert set(res) == {"persistence", "ar_ridge", "lstm"}
    for name, runs in res.items():
        for run in runs.values():
            for h in ("7", "14", "30"):
                assert 0 < run["metrics"][h]["RMSE"] < 100, (name, h)
    # the shipped no-learning baseline must reproduce the numbers quoted in the report (9.86 / 9.38 / 11.41)
    p = res["persistence"]["0"]["metrics"]
    assert abs(p["7"]["RMSE"] - 9.86) < 0.01 and abs(p["14"]["RMSE"] - 9.38) < 0.01 and abs(p["30"]["RMSE"] - 11.41) < 0.01
    assert (tmp_path / "true_test.npy").exists()
