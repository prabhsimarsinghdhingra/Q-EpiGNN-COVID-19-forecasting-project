"""Generate figures + summary tables for the report/PPT from results/results.json."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": .3})
R = json.load(open("results/results.json"))
NAMES = {"persistence": "Persistence", "ar_ridge": "AR (Ridge)", "lstm": "LSTM (no graph)", "gcn_lstm": "GCN-LSTM",
         "gat_lstm": "GAT-LSTM", "qepignn": "Q-EpiGNN (proposed)", "abl_no_amp": "Abl: no amplitude enc.",
         "abl_heat": "Abl: classical heat diffusion", "abl_no_walk": "Abl: no graph walk", "qepignn_hybrid": "Q-EpiGNN-Hybrid (extension)"}
H = (7, 14, 30)


def agg(name):
    runs = [v["metrics"] for v in R[name].values()]
    o = {}
    for h in H:
        for m in ("RMSE", "MAE", "MAPE"):
            x = np.array([r[str(h)][m] for r in runs])
            o[(h, m)] = (x.mean(), x.std())
    o["curve"] = np.mean([r["rmse_curve"] for r in runs], 0)
    o["n"] = len(runs)
    return o


A = {k: agg(k) for k in NAMES if k in R}
table = {}
for k, o in A.items():
    table[k] = {f"{m}_{h}": [round(float(o[(h, m)][0]), 3), round(float(o[(h, m)][1]), 3)] for h in H
                for m in ("RMSE", "MAE", "MAPE")}
    table[k]["n_seeds"] = o["n"]
json.dump(table, open("results/summary.json", "w"), indent=1)

# ---- Fig: RMSE vs horizon
fig, ax = plt.subplots(figsize=(6.6, 4))
sty = {"persistence": ":", "ar_ridge": ":", "lstm": "--", "gcn_lstm": "--", "gat_lstm": "--", "abl_heat": "-.", "qepignn_hybrid": "-.", "qepignn": "-"}
for k in sty:
    if k in A:
        ax.plot(range(1, 31), A[k]["curve"], sty[k], lw=2.6 if k == "qepignn" else 1.5, label=NAMES[k])
ax.set_xlabel("Forecast horizon (days)"); ax.set_ylabel("RMSE (cases / 100k / day)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/fig_rmse_horizon.png", dpi=200); plt.close()

# ---- Fig: grouped bars at h=7/14/30 for RMSE
fig, ax = plt.subplots(figsize=(7.5, 4))
ks = [k for k in ("persistence", "ar_ridge", "lstm", "gcn_lstm", "gat_lstm", "qepignn") if k in A]
w = 0.8 / len(ks)
for i, k in enumerate(ks):
    ax.bar(np.arange(3) + i * w, [A[k][(h, "RMSE")][0] for h in H], w,
           yerr=[A[k][(h, "RMSE")][1] for h in H], label=NAMES[k], capsize=2)
ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels([f"{h}-day" for h in H])
ax.set_ylabel("RMSE (cases / 100k / day)"); ax.legend(fontsize=7, ncol=2); fig.tight_layout()
fig.savefig("figures/fig_rmse_bars.png", dpi=200); plt.close()

# ---- Fig: ablation (RMSE mean over horizons)
ab = [k for k in ("qepignn", "qepignn_hybrid", "abl_no_amp", "abl_heat", "abl_no_walk", "gcn_lstm") if k in A]
fig, ax = plt.subplots(figsize=(6.6, 3.8))
vals = [np.mean([A[k][(h, "RMSE")][0] for h in H]) for k in ab]
errs = [np.mean([A[k][(h, "RMSE")][1] for h in H]) for k in ab]
ax.barh([NAMES[k] for k in ab], vals, xerr=errs, color=["#c0392b"] + ["#7f8c8d"] * (len(ab) - 1))
ax.invert_yaxis(); ax.set_xlabel("Mean RMSE over 7/14/30-day horizons"); fig.tight_layout()
fig.savefig("figures/fig_ablation.png", dpi=200); plt.close()

# ---- Fig: influence matrix + top influencers
try:
    M = np.load("results/influence_qepignn.npy"); st = np.load("results/test_meta.npz", allow_pickle=True)["states"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(M, cmap="viridis"); ax.grid(False)
    ax.set_xlabel("Target state index"); ax.set_ylabel("Source state index")
    plt.colorbar(im, label="Quantum-walk transition probability"); fig.tight_layout()
    fig.savefig("figures/fig_influence.png", dpi=200); plt.close()
    out_inf = (M.sum(0) - np.diag(M)); top = np.argsort(-out_inf)[:10]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh([st[i] for i in top][::-1], out_inf[top][::-1], color="#2c6fbb")
    ax.set_xlabel("Total inbound influence from other states"); fig.tight_layout()
    fig.savefig("figures/fig_top_influencers.png", dpi=200); plt.close()
    json.dump({st[i]: float(out_inf[i]) for i in top}, open("results/top_influencers.json", "w"), indent=1)
except Exception as e:
    print("influence fig skipped:", e)

# ---- Fig: forecast example (7-day) national total + a few states
try:
    T = np.load("results/true_test.npy"); meta = np.load("results/test_meta.npz", allow_pickle=True)
    d = meta["dates"]; st = meta["states"]
    pq = np.load("results/pred_qepignn_s0.npy"); pg = np.load("results/pred_gcn_lstm_s0.npy")
    pp = np.load("results/pred_persistence_s0.npy")
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, s in zip(axs, ("California", "Texas", "New York")):
        i = list(st).index(s); ax.plot(T[:, i, 6], "k", lw=2, label="Observed")
        ax.plot(pq[:, i, 6], "r", label="Q-EpiGNN"); ax.plot(pg[:, i, 6], "b--", label="GCN-LSTM")
        ax.plot(pp[:, i, 6], "g:", label="Persistence")
        ax.set_title(f"{s} (7-day-ahead)"); ax.set_xlabel("Test day"); ax.set_ylabel("Cases / 100k")
    axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig("figures/fig_forecast_examples.png", dpi=200); plt.close()
except Exception as e:
    print("forecast fig skipped:", e)

# ---- print markdown-ish table
for k, o in A.items():
    print(f"{NAMES[k]:32s} n={o['n']} | " + " | ".join(
        f"h{h} RMSE {o[(h,'RMSE')][0]:.2f}±{o[(h,'RMSE')][1]:.2f} MAE {o[(h,'MAE')][0]:.2f} MAPE {o[(h,'MAPE')][0]:.1f}" for h in H))
