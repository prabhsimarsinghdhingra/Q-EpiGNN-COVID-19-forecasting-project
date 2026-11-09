import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)  # scripts use paths relative to the repository root


@pytest.fixture(scope="session")
def small_graph():
    """Random connected 8-node graph with positive weights (symmetric)."""
    rng = np.random.default_rng(0)
    N = 8
    A = np.zeros((N, N))
    for i in range(N - 1):  # chain guarantees connectivity
        A[i, i + 1] = A[i + 1, i] = 1
    for _ in range(6):
        i, j = rng.integers(0, N, 2)
        if i != j:
            A[i, j] = A[j, i] = 1
    W = A * rng.uniform(0.1, 1.0, (N, N))
    W = (W + W.T) / 2
    return A, W


@pytest.fixture(scope="session")
def synthetic_data(small_graph):
    """Tiny synthetic epidemic tensor in the format expected by train.run_neural."""
    import train
    A, W = small_graph
    N, T = A.shape[0], 420
    rng = np.random.default_rng(1)
    t = np.arange(T)[:, None]
    base = 3 + 2 * np.sin(t / 25 + rng.uniform(0, 6, (1, N))) + rng.normal(0, .05, (T, N))
    X = np.stack([base, 0.3 * base, np.gradient(base, axis=0)], -1).astype(np.float32)
    tr, va, te = train.make_splits(T)
    mu, sd = X[tr[0]:tr[-1] + 1].mean((0, 1)), X[tr[0]:tr[-1] + 1].std((0, 1)) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    return (Xn, X, X[..., 0], A, W, (tr, va, te), mu, sd)
