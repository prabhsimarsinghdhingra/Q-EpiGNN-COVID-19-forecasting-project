import numpy as np
import pytest
import torch

import models as M

ALL = ["qepignn", "abl_no_amp", "abl_heat", "abl_no_walk", "gcn_lstm", "gat_lstm", "lstm", "qepignn_hybrid"]


def _spatial(small_graph, **kw):
    A, W = small_graph
    B = A * (0.1 + (W / W.max()) ** 0.25)
    return M.QuantumWalkSpatial(3, 16, M.sym_norm_laplacian(B), **kw)


def test_normalised_laplacian_spectrum(small_graph):
    A, _ = small_graph
    lam = np.linalg.eigvalsh(M.sym_norm_laplacian(A))
    assert lam.min() > -1e-9 and lam.max() < 2 + 1e-9
    assert abs(lam.min()) < 1e-9  # connected graph -> eigenvalue 0


def test_gcn_norm_symmetric(small_graph):
    A, _ = small_graph
    G = M.gcn_norm(A)
    assert np.allclose(G, G.T) and np.all(np.linalg.eigvalsh(G) <= 1 + 1e-9)


def test_quantum_walk_operator_is_unitary(small_graph):
    sp = _spatial(small_graph)
    C, S = sp.operators()
    N = C.shape[-1]
    for k in range(C.shape[0]):
        U = C[k].detach().numpy() - 1j * S[k].detach().numpy()
        assert np.allclose(U @ U.conj().T, np.eye(N), atol=1e-4), "U U^dagger must equal identity"


def test_walk_probabilities_are_doubly_stochastic(small_graph):
    sp = _spatial(small_graph)
    C, S = sp.operators()
    P = (C ** 2 + S ** 2).detach().numpy()
    assert np.allclose(P.sum(-1), 1, atol=1e-4) and np.allclose(P.sum(-2), 1, atol=1e-4)
    M_inf = sp.influence().numpy()
    assert np.allclose(M_inf.sum(1), 1, atol=1e-4) and (M_inf >= -1e-7).all()


def test_heat_kernel_is_not_unitary(small_graph):
    sp = _spatial(small_graph, walk="heat")
    K, S = sp.operators()
    assert S is None
    assert not np.allclose(K[0].detach().numpy() @ K[0].detach().numpy().T, np.eye(K.shape[-1]), atol=1e-2)


def test_amplitude_encoding_has_unit_norm(small_graph):
    sp = _spatial(small_graph, walk="none")
    x = torch.randn(4, 8, 3)
    z = sp.lift(x)
    u = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
    th = sp.phase(x)
    a, b = u * torch.cos(th), u * torch.sin(th)
    assert torch.allclose((a ** 2 + b ** 2).sum(-1), torch.ones(4, 8), atol=1e-4)


@pytest.mark.parametrize("name", ALL)
def test_forecaster_shapes_and_finite(name, small_graph):
    A, W = small_graph
    model = M.build_model(name, A, W, in_dim=3, d=16, horizon=30)
    x = torch.randn(2, 14, A.shape[0], 3)
    base = torch.randn(2, A.shape[0])
    y = model(x, base)
    assert y.shape == (2, A.shape[0], 30) and torch.isfinite(y).all()


@pytest.mark.parametrize("name", ALL)
def test_all_parameters_receive_gradients(name, small_graph):
    A, W = small_graph
    torch.manual_seed(0)
    model = M.build_model(name, A, W, in_dim=3, d=16, horizon=30)
    loss = model(torch.randn(2, 14, A.shape[0], 3), torch.randn(2, A.shape[0])).pow(2).mean()
    loss.backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not dead, f"no/invalid gradient for {dead}"


def test_walk_times_are_trainable(small_graph):
    A, W = small_graph
    model = M.build_model("qepignn", A, W, in_dim=3, d=16)
    model(torch.randn(2, 14, A.shape[0], 3), torch.zeros(2, A.shape[0])).sum().backward()
    assert model.spatial.log_t.grad is not None and model.spatial.log_t.grad.abs().sum() > 0


def test_persistence_skip_connection(small_graph):
    """With zeroed decoder weights the forecast equals the last observed value (residual design)."""
    A, W = small_graph
    model = M.build_model("gcn_lstm", A, W, in_dim=3, d=16)
    for p in model.dec.parameters():
        torch.nn.init.zeros_(p)
    base = torch.randn(2, A.shape[0])
    y = model(torch.randn(2, 14, A.shape[0], 3), base)
    assert torch.allclose(y, base.unsqueeze(-1).expand_as(y))


def test_unknown_model_raises(small_graph):
    A, W = small_graph
    with pytest.raises(ValueError):
        M.build_model("does_not_exist", A, W, in_dim=3)


def test_hybrid_has_both_branches_and_valid_influence(small_graph):
    sp = _spatial(small_graph, walk="hybrid")
    assert sp.out.in_features == 4 * 4 * 16 + 1 + 16  # 3K quantum blocks + K heat blocks + magnitude + skip
    M_inf = sp.influence().numpy()
    assert np.allclose(M_inf.sum(1), 1, atol=1e-4)
