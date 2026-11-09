"""Split-conformal prediction intervals for the forecasters (computed in log1p space, per horizon).

static  : one quantile from the validation residuals (classic split conformal)
rolling : quantile from the most recent `window` forecast origins whose h-day-ahead target is already observed
          at the time of forecasting (valid online recalibration; helps under distribution shift)
Output: results/uncertainty.json and figures/fig_uncertainty.png
"""
import json

import numpy as np

HORIZONS = (7, 14, 30)


def conformal_quantile(res, alpha):
    """Finite-sample-corrected (1-alpha) empirical quantile of absolute residuals."""
    res = np.sort(np.asarray(res).ravel())
    n = len(res)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return res[k - 1]


def intervals_static(val_res_h, pred_log_h, alpha):
    q = conformal_quantile(val_res_h, alpha)
    return pred_log_h - q, pred_log_h + q


def intervals_rolling(res_seq_h, n_val, pred_log_h, h, alpha, window=120):
    """res_seq_h: (Bv+Bt, N) abs residuals in time order (validation windows first, then test windows).
    For test window i (sequence position n_val+i) only windows j <= n_val+i-h have an observed target."""
    if n_val < h:
        raise ValueError(f"need at least h={h} calibration windows before the test period, got {n_val}")
    lo, hi = np.empty_like(pred_log_h), np.empty_like(pred_log_h)
    for i in range(len(pred_log_h)):
        last = n_val + i - h
        q = conformal_quantile(res_seq_h[max(0, last - window + 1):last + 1], alpha)
        lo[i], hi[i] = pred_log_h[i] - q, pred_log_h[i] + q
    return lo, hi


def evaluate(lo_log, hi_log, true_cases):
    lo, hi = np.expm1(np.clip(lo_log, 0, 12)), np.expm1(np.clip(hi_log, 0, 12))
    return dict(coverage=float(((true_cases >= lo) & (true_cases <= hi)).mean()), width=float((hi - lo).mean()))


def main(res="results/", alpha=0.1, models=("qepignn", "abl_heat", "qepignn_hybrid")):
    import train
    X, c100, A, W, dates, states = train.load()
    tr, va, te = train.make_splits(len(X))
    ylog_val = train.targets(X[..., 0], va)
    ylog_te = train.targets(X[..., 0], te)
    true_c = np.load(res + "true_test.npy")
    out = {}
    for m in models:
        try:
            pv = np.load(res + f"predval_{m}_s0.npy")
            pt = np.log1p(np.load(res + f"pred_{m}_s0.npy"))
        except FileNotFoundError:
            continue
        out[m] = {}
        for h in HORIZONS:
            rv = np.abs(ylog_val[..., h - 1] - pv[..., h - 1])
            rt = np.abs(ylog_te[..., h - 1] - pt[..., h - 1])
            ls, hs = intervals_static(rv, pt[..., h - 1], alpha)
            lr, hr = intervals_rolling(np.concatenate([rv, rt]), len(rv), pt[..., h - 1], h, alpha)
            out[m][f"h{h}"] = dict(static=evaluate(ls, hs, true_c[..., h - 1]), rolling=evaluate(lr, hr, true_c[..., h - 1]))
            s, r = out[m][f"h{h}"]["static"], out[m][f"h{h}"]["rolling"]
            print(f"{m:15s} h{h:2d} target {100*(1-alpha):.0f}% | static cov {s['coverage']:.3f} width {s['width']:.1f}"
                  f" | rolling cov {r['coverage']:.3f} width {r['width']:.1f}")
    json.dump(out, open(res + "uncertainty.json", "w"), indent=1)

    # figure: California 7-day-ahead forecast with rolling-conformal band for the best available model
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m = "abl_heat" if "abl_heat" in out else next(iter(out))
        pv, pt = np.load(res + f"predval_{m}_s0.npy"), np.log1p(np.load(res + f"pred_{m}_s0.npy"))
        h = 14
        rv = np.abs(ylog_val[..., h - 1] - pv[..., h - 1]); rt = np.abs(ylog_te[..., h - 1] - pt[..., h - 1])
        lo, hi = intervals_rolling(np.concatenate([rv, rt]), len(rv), pt[..., h - 1], h, alpha)
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, s in zip(axs, ("California", "Texas", "New York")):
            i = list(states).index(s)
            ax.fill_between(range(len(te)), np.expm1(np.clip(lo[:, i], 0, 12)), np.expm1(np.clip(hi[:, i], 0, 12)), alpha=.25, color="tab:red", label="90% interval")
            ax.plot(true_c[:, i, h - 1], "k", lw=1.8, label="Observed"); ax.plot(np.expm1(pt[:, i, h - 1]), "r", label="Forecast")
            ax.set_title(f"{s} ({h}-day-ahead)"); ax.set_xlabel("Test day"); ax.set_ylabel("Cases / 100k"); ax.grid(alpha=.3)
        axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/fig_uncertainty.png", dpi=200); plt.close()
    except Exception as e:  # plotting is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
