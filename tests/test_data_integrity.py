"""Integrity checks on the processed dataset shipped in data/processed (skipped if absent)."""
import os

import numpy as np
import pytest

PATH = "data/processed/covid_us_states.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(PATH), reason="processed data not present")


@pytest.fixture(scope="module")
def z():
    return np.load(PATH, allow_pickle=True)


def test_shapes_and_alignment(z):
    T, N = z["cases"].shape
    assert N == 51 and T > 1000
    assert z["deaths"].shape == (T, N) and len(z["dates"]) == T and len(z["states"]) == N
    assert z["A"].shape == (N, N) and z["pop"].shape == (N,)


def test_no_nans_or_negatives(z):
    for k in ("cases", "deaths", "raw_cases", "pop"):
        assert np.isfinite(z[k]).all() and (z[k] >= 0).all(), k


def test_dates_strictly_increasing_daily(z):
    d = np.array(z["dates"], dtype="datetime64[D]")
    assert (np.diff(d).astype(int) == 1).all()


def test_totals_match_jhu_scale(z):
    assert 9.5e7 < z["raw_cases"].sum() < 1.1e8   # ~102.7 M confirmed cases in the 50 states + DC
    assert 4e8 < z["pop"].sum() < 3.5e8 * 1.1 or 3e8 < z["pop"].sum() < 3.6e8


def test_graph_is_symmetric_and_connected(z):
    A = z["A"]
    assert np.allclose(A, A.T)
    from numpy.linalg import matrix_power
    reach = (matrix_power(A + np.eye(len(A)), len(A)) > 0)
    assert reach.all(), "state graph must be connected"
