import numpy as np

import preprocess as pp


def _fake_centroids():
    rng = np.random.default_rng(0)
    return np.stack([rng.uniform(25, 48, 51), rng.uniform(-124, -70, 51)], 1), rng.uniform(5e5, 4e7, 51)


def test_state_list_is_complete():
    assert len(pp.STATES) == 51 and len(set(pp.STATES)) == 51
    assert "District of Columbia" in pp.STATES and set(pp.ABBR.values()) == set(pp.STATES)


def test_borders_reference_valid_states_and_no_duplicates():
    pairs = [frozenset(b.split("-")) for b in pp.BORDERS]
    assert all(len(p) == 2 and all(x in pp.ABBR for x in p) for p in pairs)
    assert len(pairs) == len(set(pairs))


def test_graph_properties():
    cent, P = _fake_centroids()
    A, W, G = pp.build_graph(cent, P)
    assert A.shape == W.shape == G.shape == (51, 51)
    assert np.allclose(A, A.T) and np.allclose(W, W.T) and np.allclose(G, G.T)
    assert np.all(np.diag(A) == 0) and np.all(np.diag(G) == 0)
    assert set(np.unique(A)) <= {0.0, 1.0}
    assert A.sum(1).min() >= 1, "every node (incl. Alaska/Hawaii) must have a neighbour"
    assert 0 < W.max() <= 1 + 1e-12 and np.all(W[A == 0] == 0) and np.all(W[A == 1] > 0)


def test_known_adjacencies():
    cent, P = _fake_centroids()
    A, _, _ = pp.build_graph(cent, P)
    idx = {s: i for i, s in enumerate(pp.STATES)}
    assert A[idx["California"], idx["Nevada"]] == 1
    assert A[idx["Texas"], idx["Oklahoma"]] == 1
    assert A[idx["Maine"], idx["New Hampshire"]] == 1
    assert A[idx["California"], idx["Florida"]] == 0
    assert A[idx["District of Columbia"], idx["Virginia"]] == 1


def test_isolated_states_connect_to_two_nearest():
    cent, P = _fake_centroids()
    A, _, _ = pp.build_graph(cent, P)
    for s in ("Alaska", "Hawaii"):
        i = pp.STATES.index(s)
        assert A[i].sum() == 2
