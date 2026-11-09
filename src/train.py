"""Train / evaluate Q-EpiGNN, baselines and ablations on JHU CSSE US state-level COVID-19 data.

Usage:  python src/train.py --models qepignn gcn_lstm --seeds 0 1 2 --epochs 100
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from models import build_model

L, H, HORIZONS = 14, 30, (7, 14, 30)
START = "2020-03-01"
RES = "results/"


# --------------------------------------------------------------------------- data
def load():
    z = np.load("data/processed/covid_us_states.npz", allow_pickle=True)
    dates = z["dates"].astype(str)
    s = int(np.argmax(dates >= START))
    pop = z["pop"]
    c100 = z["cases"][s:] / pop * 1e5  # (T,N) 7-day-avg daily new cases per 100k
    d100 = z["deaths"][s:] / pop * 1e5
    lc, ld = np.log1p(c100), np.log1p(d100)
    trend = lc - np.roll(lc, 7, axis=0)
    trend[:7] = 0
    X = np.stack([lc, ld, trend], -1).astype(np.float32)  # (T,N,3)
    return X, c100.astype(np.float32), z["A"], z["W"], dates[s:], z["states"]


def make_splits(T):
    origins = np.arange(L - 1, T - H)  # t = last observed day, targets t+1..t+H
    n = len(origins)
    a, b = int(0.70 * n), int(0.80 * n)
    tr = origins[:a - H]  # purge H days so no training target overlaps the validation period
    va = origins[a:b - H]
    te = origins[b:]
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"series too short (T={T}) for a {L}-day window and {H}-day horizon: empty split")
    return tr, va, te


def windows(X, idx):
    return np.stack([X[t - L + 1:t + 1] for t in idx])  # (B,L,N,F)


def targets(lc, idx):
    return np.stack([lc[t + 1:t + 1 + H].T for t in idx])  # (B,N,H) log1p(cases/100k)


# --------------------------------------------------------------------------- metrics
def metrics(pred_c, true_c):
    """pred_c,true_c: (B,N,H) cases per 100k. Returns dict per horizon."""
    out = {}
    for h in HORIZONS:
        p, t = pred_c[..., h - 1].ravel(), true_c[..., h - 1].ravel()
        m = t >= 1.0  # MAPE only where truth >= 1 case/100k/day (avoids division blow-up at ~0)
        out[h] = dict(RMSE=float(np.sqrt(np.mean((p - t) ** 2))), MAE=float(np.mean(np.abs(p - t))),
                      MAPE=float(100 * np.mean(np.abs(p[m] - t[m]) / t[m])))
    out["rmse_curve"] = [float(np.sqrt(np.mean((pred_c[..., k] - true_c[..., k]) ** 2))) for k in range(H)]
    return out


def to_cases(log_pred):
    return np.expm1(np.clip(log_pred, 0, 12))


# --------------------------------------------------------------------------- neural training
def run_neural(name, seed, data, epochs, lr, d, bs=32, patience=20, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xn, Xraw, Y, A, W, (tr, va, te), mu, sd = data
    model = build_model(name, A, W, in_dim=Xn.shape[-1], d=d, horizon=H)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    lossf = nn.HuberLoss(delta=0.5)

    def batch(idx):
        x = torch.tensor(windows(Xn, idx))
        base = torch.tensor(Xraw[idx, :, 0])
        return x, base, torch.tensor(targets(Xraw[..., 0], idx))

    def predict(idx):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                x, base, _ = batch(idx[i:i + 128])
                outs.append(model(x, base).numpy())
        return np.concatenate(outs)

    best, best_state, bad = 1e9, None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            x, base, y = batch(perm[i:i + bs])
            opt.zero_grad()
            loss = lossf(model(x, base), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = float(np.mean((predict(va) - targets(Xraw[..., 0], va)) ** 2))
        sched.step(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and ep % 5 == 0:
            print(f"  ep{ep:3d} val_mse(log)={vl:.4f} best={best:.4f} {time.time()-t0:.0f}s", flush=True)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    pred = predict(te)
    extra = {"val_pred_log": predict(va)}
    if hasattr(model.spatial, "influence"):
        extra["influence"] = model.spatial.influence().numpy()
        if hasattr(model.spatial, "log_t"):
            extra["walk_times"] = model.spatial.log_t.exp().detach().numpy().tolist()
            extra["gate"] = model.spatial.gate.detach().numpy().tolist()
    n_par = sum(p.numel() for p in model.parameters())
    return pred, dict(val_mse=best, epochs=ep + 1, seconds=time.time() - t0, params=n_par), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qepignn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)

    X, c100, A, Wm, dates, states = load()
    tr, va, te = make_splits(len(X))
    mu, sd = X[tr[0]:tr[-1] + 1].mean((0, 1)), X[tr[0]:tr[-1] + 1].std((0, 1)) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    data = (Xn, X, X[..., 0], A, Wm, (tr, va, te), mu, sd)
    print(f"T={len(X)} | train {len(tr)} val {len(va)} test {len(te)} | test origins {dates[te[0]]} -> {dates[te[-1]]}")
    true_c = to_cases(targets(X[..., 0], te))

    os.makedirs(RES, exist_ok=True)
    path = RES + a.out
    res = json.load(open(path)) if os.path.exists(path) else {}

    def save(name, key, met, info=None):
        res.setdefault(name, {})[key] = dict(metrics=met, info=info or {})
        json.dump(res, open(path, "w"))

    for name in a.models:
        if name == "persistence":
            pred = to_cases(np.repeat(X[te][:, :, 0][..., None], H, -1))
            save(name, "0", metrics(pred, true_c))
            np.save(RES + f"pred_{name}_s0.npy", pred)
            print(name, {h: round(metrics(pred, true_c)[h]["RMSE"], 2) for h in HORIZONS}, flush=True)
        elif name == "ar_ridge":
            def feats(idx):
                w = windows(Xn, idx)  # (B,L,N,F)
                return w.transpose(0, 2, 1, 3).reshape(len(idx) * w.shape[2], -1)
            def tgt(idx):
                y = targets(X[..., 0], idx) - X[idx][:, :, 0][..., None]
                return y.reshape(-1, H)
            m = Ridge(alpha=10.0).fit(feats(tr), tgt(tr))
            d = m.predict(feats(te)).reshape(len(te), -1, H)
            pred = to_cases(X[te][:, :, 0][..., None] + d)
            save(name, "0", metrics(pred, true_c))
            np.save(RES + f"pred_{name}_s0.npy", pred)
            print(name, {h: round(metrics(pred, true_c)[h]["RMSE"], 2) for h in HORIZONS}, flush=True)
        else:
            for s in a.seeds:
                pred_log, info, extra = run_neural(name, s, data, a.epochs, a.lr, a.d, verbose=a.verbose)
                pred = to_cases(pred_log)
                met = metrics(pred, true_c)
                save(name, str(s), met, info)
                if s == a.seeds[0]:
                    np.save(RES + f"pred_{name}_s{s}.npy", pred)
                    if "influence" in extra:
                        np.save(RES + f"influence_{name}.npy", extra["influence"])
                        json.dump({k: v for k, v in extra.items() if k not in ("influence", "val_pred_log")},
                                  open(RES + f"walk_{name}.json", "w"))
                    np.save(RES + f"predval_{name}_s{s}.npy", extra["val_pred_log"])  # for conformal intervals
                print(f"{name} seed{s} | " + " ".join(f"h{h}: RMSE {met[h]['RMSE']:.2f} MAE {met[h]['MAE']:.2f} "
                      f"MAPE {met[h]['MAPE']:.1f}" for h in HORIZONS) +
                      f" | ep {info['epochs']} {info['seconds']:.0f}s params {info['params']}", flush=True)
    np.save(RES + "true_test.npy", true_c)
    np.savez(RES + "test_meta.npz", dates=dates[te], states=states)


if __name__ == "__main__":
    main()
