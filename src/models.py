"""Q-EpiGNN and baseline spatiotemporal forecasters (PyTorch, CPU friendly).

All learned models share the same skeleton so that comparisons are fair:
    spatial module (per day)  ->  node-wise LSTM over the input window  ->  MLP decoder (30-day horizon)
and predict a residual on top of the last observed log-incidence (persistence skip connection).
Only the *spatial module* differs between models.

"Quantum-inspired" = classical simulation of quantum ideas (no quantum hardware):
  * amplitude (+phase) encoding : node features -> unit-norm complex amplitude vector psi = u * exp(i*theta)
  * continuous-time quantum walk: psi' = exp(-i H t) psi, H = normalised graph Laplacian (unitary diffusion)
  * interference read-out       : [Re psi', Im psi', |psi'|^2] at several walk times t_k (multi-scale)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- graph helpers
def sym_norm_laplacian(B: np.ndarray) -> np.ndarray:
    d = B.sum(1)
    dinv = np.where(d > 0, d ** -0.5, 0.0)
    return np.eye(len(B)) - dinv[:, None] * B * dinv[None]


def gcn_norm(A: np.ndarray) -> np.ndarray:
    At = A + np.eye(len(A))
    dinv = At.sum(1) ** -0.5
    return dinv[:, None] * At * dinv[None]


# --------------------------------------------------------------------------- spatial modules
class QuantumWalkSpatial(nn.Module):
    """Amplitude encoding + multi-scale continuous-time quantum walk (the proposed module).

    walk: "quantum"  -> unitary U(t)=exp(-iHt)        (Q-EpiGNN)
          "hybrid"   -> unitary branch + heat branch  (extension: lets the read-out choose the mix)
          "heat"     -> non-unitary exp(-Ht)           (classical diffusion ablation)
          "none"     -> identity (no message passing)  (ablation)
    enc : "amplitude" -> unit-norm amplitudes with learned phase (+ magnitude channel)
          "linear"    -> plain linear lift, zero imaginary part (ablation)
    """

    def __init__(self, in_dim, d, H_np, times=(0.5, 1.0, 2.0, 4.0), walk="quantum", enc="amplitude"):
        super().__init__()
        self.d, self.walk, self.enc, self.K = d, walk, enc, len(times)
        lam, V = np.linalg.eigh(H_np)  # H real symmetric -> exact spectral decomposition
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float32))
        self.register_buffer("V", torch.tensor(V, dtype=torch.float32))
        self.log_t = nn.Parameter(torch.log(torch.tensor(times, dtype=torch.float32)))  # learnable walk times
        self.lift = nn.Linear(in_dim, d)
        self.phase = nn.Linear(in_dim, d)
        if enc != "amplitude":
            # unused in the linear-encoding ablation; kept (frozen) so random initialisation, and hence the
            # reported results, stay identical to the runs in results/results.json
            self.phase.requires_grad_(False)
        self.gate = nn.Parameter(torch.ones(self.K))  # per-scale importance (used for interpretability)
        if walk == "none":  # no propagation -> walk times / gates are unused
            self.log_t.requires_grad_(False)
            self.gate.requires_grad_(False)
        n_feat = {"quantum": 3 * self.K, "heat": self.K, "none": 1, "hybrid": 4 * self.K}[walk]
        self.out = nn.Linear(n_feat * d + (1 if enc == "amplitude" else 0) + d, d)
        self.norm = nn.LayerNorm(d)

    def operators(self):
        t = self.log_t.exp()  # (K,)
        ang = self.lam[None, :] * t[:, None]  # (K,N)
        V = self.V
        if self.walk in ("quantum", "hybrid"):
            C = torch.einsum("ik,tk,jk->tij", V, torch.cos(ang), V)
            S = torch.einsum("ik,tk,jk->tij", V, torch.sin(ang), V)
            return C, S
        K_heat = torch.einsum("ik,tk,jk->tij", V, torch.exp(-ang), V)
        return K_heat, None

    def influence(self):
        """Interpretable spread-influence matrix  M_ij = sum_k |g_k| * P_k(i->j)  (rows sum to ~1)."""
        with torch.no_grad():
            C, S = self.operators()
            P = C ** 2 + S ** 2 if self.walk in ("quantum", "hybrid") else C.abs()
            g = self.gate.abs() / self.gate.abs().sum()
            return (g[:, None, None] * P).sum(0)

    def forward(self, x):  # x: (M, N, F)
        z = self.lift(x)
        if self.enc == "amplitude":
            r = z.norm(dim=-1, keepdim=True)
            u = z / (r + 1e-6)  # amplitude encoding: unit-norm state vector
            th = self.phase(x)
            a, b = u * torch.cos(th), u * torch.sin(th)
            mag = [r]
        else:
            a, b, mag = z, torch.zeros_like(z), []
        if self.walk == "none":
            feats = [a]
        else:
            C, S = self.operators()
            feats = []
            if self.walk == "hybrid":  # dissipative branch: heat kernel exp(-tH) applied to the real amplitude
                Kh = torch.einsum("ik,tk,jk->tij", self.V, torch.exp(-self.lam[None, :] * self.log_t.exp()[:, None]), self.V)
                feats += [self.gate[k] * torch.einsum("ij,mjd->mid", Kh[k], a) for k in range(self.K)]
            for k in range(self.K):
                if self.walk in ("quantum", "hybrid"):
                    re = torch.einsum("ij,mjd->mid", C[k], a) + torch.einsum("ij,mjd->mid", S[k], b)
                    im = torch.einsum("ij,mjd->mid", C[k], b) - torch.einsum("ij,mjd->mid", S[k], a)
                    feats += [self.gate[k] * re, self.gate[k] * im, self.gate[k] * (re ** 2 + im ** 2)]
                else:
                    feats += [self.gate[k] * torch.einsum("ij,mjd->mid", C[k], a)]
        h = self.out(torch.cat(feats + mag + [z], -1))
        return F.gelu(self.norm(h))


class GCNSpatial(nn.Module):
    def __init__(self, in_dim, d, A_np):
        super().__init__()
        self.register_buffer("Ah", torch.tensor(gcn_norm(A_np), dtype=torch.float32))
        self.l1, self.l2 = nn.Linear(in_dim, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = F.gelu(torch.einsum("ij,mjf->mif", self.Ah, self.l1(x)))
        h = torch.einsum("ij,mjf->mif", self.Ah, self.l2(h))
        return F.gelu(self.norm(h))


class GATSpatial(nn.Module):
    def __init__(self, in_dim, d, A_np, heads=4):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.register_buffer("mask", torch.tensor(A_np + np.eye(len(A_np)) > 0))
        self.W = nn.Linear(in_dim, d)
        self.a_src = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.W2 = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        M, N, _ = x.shape
        z = self.W(x).view(M, N, self.h, self.dk)
        es = (z * self.a_src).sum(-1)  # (M,N,h)
        ed = (z * self.a_dst).sum(-1)
        e = F.leaky_relu(es.unsqueeze(2) + ed.unsqueeze(1), 0.2)  # (M,N,N,h)
        e = e.masked_fill(~self.mask[None, :, :, None], -1e9)
        att = torch.softmax(e, dim=2)
        out = torch.einsum("mijh,mjhd->mihd", att, z).reshape(M, N, -1)
        return F.gelu(self.norm(self.W2(F.gelu(out))))


class NoGraphSpatial(nn.Module):
    def __init__(self, in_dim, d):
        super().__init__()
        self.l1, self.l2, self.norm = nn.Linear(in_dim, d), nn.Linear(d, d), nn.LayerNorm(d)

    def forward(self, x):
        return F.gelu(self.norm(self.l2(F.gelu(self.l1(x)))))


# --------------------------------------------------------------------------- full forecaster
class Forecaster(nn.Module):
    def __init__(self, spatial: nn.Module, d=32, horizon=30):
        super().__init__()
        self.spatial = spatial
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, horizon))

    def forward(self, xn, base):
        """xn: (B,L,N,F) standardised inputs; base: (B,N) last observed log-incidence."""
        B, L, N, Fd = xn.shape
        s = self.spatial(xn.reshape(B * L, N, Fd)).reshape(B, L, N, -1)
        s = s.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        o, _ = self.lstm(s)
        delta = self.dec(o[:, -1]).reshape(B, N, -1)  # (B,N,H)
        return base.unsqueeze(-1) + delta  # predicted log1p(cases / 100k)


def build_model(name, A, W, in_dim, d=32, horizon=30, gamma=0.25):
    # Hamiltonian edge weights: border adjacency modulated by (normalised) gravity mobility proxy
    B = A * (0.1 + (W / (W.max() + 1e-12)) ** gamma)
    H = sym_norm_laplacian(B)
    if name == "qepignn":
        sp = QuantumWalkSpatial(in_dim, d, H, walk="quantum", enc="amplitude")
    elif name == "abl_no_amp":
        sp = QuantumWalkSpatial(in_dim, d, H, walk="quantum", enc="linear")
    elif name == "abl_heat":
        sp = QuantumWalkSpatial(in_dim, d, H, walk="heat", enc="amplitude")
    elif name == "qepignn_hybrid":  # post-hoc extension: unitary + dissipative branches (see report Sec. 4.4)
        sp = QuantumWalkSpatial(in_dim, d, H, walk="hybrid", enc="amplitude")
    elif name == "abl_no_walk":
        sp = QuantumWalkSpatial(in_dim, d, H, walk="none", enc="amplitude")
    elif name == "gcn_lstm":
        sp = GCNSpatial(in_dim, d, A)
    elif name == "gat_lstm":
        sp = GATSpatial(in_dim, d, A)
    elif name == "lstm":
        sp = NoGraphSpatial(in_dim, d)
    else:
        raise ValueError(name)
    return Forecaster(sp, d, horizon)
