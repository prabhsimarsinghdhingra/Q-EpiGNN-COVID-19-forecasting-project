"""Extra report figures: state graph, dataset overview, architecture diagram."""
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.size": 10})
z = np.load("data/processed/covid_us_states.npz", allow_pickle=True)
A, cent, pop, st = z["A"], z["centroid"], z["pop"], z["states"]
# --- graph
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for i in range(len(A)):
    for j in range(i + 1, len(A)):
        if A[i, j]: ax.plot([cent[i, 1], cent[j, 1]], [cent[i, 0], cent[j, 0]], color="#9aa5b1", lw=.8, zorder=1)
ax.scatter(cent[:, 1], cent[:, 0], s=pop / 1.2e5, c="#2c6fbb", alpha=.8, zorder=2)
for i, s in enumerate(st):
    if pop[i] > 9e6: ax.annotate(s, (cent[i, 1], cent[i, 0]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlim(-130, -65); ax.set_ylim(23, 52); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
fig.tight_layout(); fig.savefig("figures/fig_graph.png", dpi=200); plt.close()
# --- dataset overview
d = np.array(z["dates"], dtype="datetime64[D]"); tot = z["cases"].sum(1); dt = z["deaths"].sum(1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(d, tot / 1e3, color="#c0392b"); ax[0].set_ylabel("Daily new cases (thousands, 7-day avg)")
ax[1].plot(d, dt, color="#34495e"); ax[1].set_ylabel("Daily deaths (7-day avg)")
for a in ax:
    a.grid(alpha=.3); a.set_xlabel("Date"); a.axvline(np.datetime64("2022-07-10"), ls="--", c="k", lw=1)
ax[0].text(np.datetime64("2022-07-20"), tot.max() / 1e3 * .8, "test period", fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_dataset.png", dpi=200); plt.close()
# --- architecture
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 36)
boxes = [("Input window\n14 days x 51 states\n(log cases, deaths,\n7-day trend)", 1),
         ("Amplitude +\nphase encoding\n$\\psi=u\\,e^{i\\theta}$,\n$\\|u\\|=1$", 17),
         ("Multi-scale\nquantum walk\n$U(t_k)=e^{-iHt_k}$\n(K = 4 scales)", 33),
         ("Interference\nread-out\n[Re, Im, $|\\psi|^2$]\n+ magnitude", 49),
         ("Node-wise\nLSTM over\n14 days", 65),
         ("MLP decoder\n+ persistence skip\n30-day forecast", 81)]
for t, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 9), 12.6, 18, boxstyle="round,pad=0.4", fc="#eaf2fb", ec="#2c6fbb", lw=1.4))
    ax.text(x + 6.3, 18, t, ha="center", va="center", fontsize=8)
for i in range(len(boxes) - 1):
    ax.add_patch(FancyArrowPatch((boxes[i][1] + 13.4, 18), (boxes[i + 1][1] - .8, 18), arrowstyle="-|>", mutation_scale=12, color="k"))
ax.text(41, 3.5, "State graph (borders + gravity weights) -> Hamiltonian H = I - D$^{-1/2}$BD$^{-1/2}$ (fixed); walk times t$_k$ learnable",
        ha="center", fontsize=8, style="italic")
ax.add_patch(FancyArrowPatch((41, 6), (41, 8.3), arrowstyle="-|>", mutation_scale=10, color="#c0392b"))
fig.tight_layout(); fig.savefig("figures/fig_architecture.png", dpi=200); plt.close()
