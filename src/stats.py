"""Paired moving-block bootstrap on test-window errors: is model A really better than model B?

Squared errors of consecutive forecast origins are strongly autocorrelated, so windows are resampled in
blocks (default 14 days) instead of independently.  Output: results/significance.json
"""
import json

import numpy as np

HORIZONS = (7, 14, 30)


def window_sq_err(pred, true, h):
    """Mean squared error per test window (averaged over states) at horizon h. pred,true: (B,N,H)."""
    return ((pred[..., h - 1] - true[..., h - 1]) ** 2).mean(1)


def block_bootstrap_rmse_diff(e_a, e_b, block=14, n_boot=4000, seed=0):
    """Return (observed RMSE_A - RMSE_B, 95% CI low, high, two-sided p) using paired block resampling."""
    rng = np.random.default_rng(seed)
    n = len(e_a)
    obs = np.sqrt(e_a.mean()) - np.sqrt(e_b.mean())
    nb = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        idx = (starts[:, None] + np.arange(block)[None]).ravel()[:n]
        diffs[i] = np.sqrt(e_a[idx].mean()) - np.sqrt(e_b[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided percentile bootstrap p-value (consistent with the percentile CI above)
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) + 1.0 / n_boot))
    return float(obs), float(lo), float(hi), p


PAIRS = [("qepignn", "persistence"), ("qepignn", "gcn_lstm"), ("qepignn", "lstm"), ("abl_heat", "persistence"),
         ("abl_heat", "qepignn"), ("qepignn_hybrid", "persistence"), ("qepignn_hybrid", "qepignn"),
         ("qepignn_hybrid", "abl_heat")]


def main(res="results/"):
    true = np.load(res + "true_test.npy")
    out = {}
    for a, b in PAIRS:
        try:
            pa, pb = np.load(res + f"pred_{a}_s0.npy"), np.load(res + f"pred_{b}_s0.npy")
        except FileNotFoundError:
            continue
        for h in HORIZONS:
            d, lo, hi, p = block_bootstrap_rmse_diff(window_sq_err(pa, true, h), window_sq_err(pb, true, h))
            out[f"{a}__vs__{b}__h{h}"] = dict(diff=d, ci_low=lo, ci_high=hi, p=p, significant=bool(lo > 0 or hi < 0))
            print(f"{a:15s} - {b:15s} h{h:2d}: dRMSE {d:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p={p:.3f}")
    json.dump(out, open(res + "significance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
