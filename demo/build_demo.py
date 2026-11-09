"""Build a self-contained demo page (demo/qepignn_demo.html) from the saved test-set predictions."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
import train  # noqa: E402
import uncertainty as U  # noqa: E402

R = "results/"
MODELS = ["persistence", "lstm", "gcn_lstm", "gat_lstm", "qepignn", "abl_heat", "qepignn_hybrid"]
NAMES = {"persistence": "Persistence", "lstm": "LSTM", "gcn_lstm": "GCN-LSTM", "gat_lstm": "GAT-LSTM",
         "qepignn": "Q-EpiGNN", "abl_heat": "Heat-diffusion variant", "qepignn_hybrid": "Q-EpiGNN-Hybrid"}
HOR = (7, 14, 30)
meta = np.load(R + "test_meta.npz", allow_pickle=True)
true = np.load(R + "true_test.npy")
X, c100, A, W, dates, states = train.load()
tr, va, te = train.make_splits(len(X))
ylog_val, ylog_te = train.targets(X[..., 0], va), train.targets(X[..., 0], te)
rd = lambda a, k=2: np.round(a, k).tolist()  # noqa: E731
pred = {m: {str(h): rd(np.load(R + f"pred_{m}_s0.npy")[..., h - 1].T) for h in HOR} for m in MODELS}
truth = {str(h): rd(true[..., h - 1].T) for h in HOR}
band_m = "abl_heat"
pv, pt = np.load(R + f"predval_{band_m}_s0.npy"), np.log1p(np.load(R + f"pred_{band_m}_s0.npy"))
band = {}
for h in HOR:
    rv = np.abs(ylog_val[..., h - 1] - pv[..., h - 1]); rt = np.abs(ylog_te[..., h - 1] - pt[..., h - 1])
    lo, hi = U.intervals_rolling(np.concatenate([rv, rt]), len(rv), pt[..., h - 1], h, 0.1)
    band[str(h)] = dict(lo=rd(np.expm1(np.clip(lo, 0, 12)).T, 1), hi=rd(np.expm1(np.clip(hi, 0, 12)).T, 1))
S = json.load(open(R + "summary.json"))
overall = {m: {f"{k}_{h}": S[m][f"{k}_{h}"][0] for k in ("RMSE", "MAPE") for h in HOR} for m in MODELS}
data = dict(dates=[str(d) for d in meta["dates"]], states=[str(s) for s in meta["states"]], horizons=list(HOR),
            models=MODELS, names=NAMES, truth=truth, pred=pred, band=band, band_model=band_m, overall=overall,
            influence=rd(np.load(R + "influence_qepignn.npy"), 4))
html = open("demo/template.html").read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
open("demo/qepignn_demo.html", "w").write(html)
print("demo built:", round(len(html) / 1e6, 2), "MB")
