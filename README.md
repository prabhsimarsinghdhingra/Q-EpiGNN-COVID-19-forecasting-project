# Q-EpiGNN: Quantum-Inspired Graph Neural Network for COVID-19 Spread Forecasting

![CI](https://github.com/<YOUR_USERNAME>/<YOUR_REPO>/actions/workflows/ci.yml/badge.svg)

MIT Bengaluru (MAHE) - CSE Minor Specialization Mini Project (CSE 4191)

Team: Somyajeet Arukh (235805412), Prabhsimar Singh Dhingra (235805306), Ritam Santra (235805374)
Guide: Dr. Shreya Banerjee

## What it is
A spatiotemporal GNN that forecasts daily COVID-19 incidence (cases / 100k) for the 50 US states + DC,
30 days ahead. The spatial module is *quantum-inspired* (simulated classically, no quantum hardware):
* amplitude + phase encoding of node features (unit-norm complex amplitudes)
* multi-scale continuous-time quantum walk  U(t) = exp(-iHt)  on the state-border graph (H = normalised Laplacian,
  edge weights from a gravity mobility proxy), with learnable walk times
* interference read-out [Re, Im, |.|^2]  followed by a node-wise LSTM and MLP decoder.
Also interpretable: |U_ij(t)|^2 gives a state-to-state spread-influence matrix.

## Data (public)
JHU CSSE COVID-19 repository (Dong, Du, Gardner, Lancet Infect Dis 2020), county-level US time series
aggregated to 51 state nodes, 2020-03-01 -> 2023-03-09. `./download_data.sh` fetches it.

## Reproduce
    pip install -r requirements.txt
    ./run_all.sh          # download -> preprocess -> train all models/ablations (3 seeds) -> figures
Single model: `python src/train.py --models qepignn --seeds 0 --epochs 40`

## Layout
src/preprocess.py  data -> state tensor + graph | src/models.py  Q-EpiGNN, GCN/GAT-LSTM, LSTM, ablations
src/train.py  training/evaluation | src/figures.py  figures/tables | results/  metrics, predictions, logs

## Testing and CI
    pip install -r requirements-dev.txt     # pytest, pytest-cov, ruff (+ torch, see requirements.txt)
    make test        # 55 unit/integration tests, ~10 s on CPU, ~81 % coverage of src/ (models.py 100 %)
    make lint        # ruff
    make smoke       # 2-epoch end-to-end run on the shipped processed data

What the tests check (tests/):
* **models** - quantum-walk operator is unitary (U U^dagger = I), walk probabilities are doubly stochastic,
  amplitude states have unit norm, heat kernel is *not* unitary, every model has correct output shape, finite values and
  gradients, walk times are trainable, the persistence skip connection holds
* **preprocessing** - 51 states, valid/unique borders, graph symmetric, no isolated nodes, known adjacencies
* **training** - chronological split with no target leakage, window/target alignment, metric formulas (RMSE/MAE/MAPE),
  determinism for a fixed seed, CLI end to end (persistence reproduces the 9.86 / 9.38 / 11.41 RMSE quoted in the report)
* **stats / uncertainty** - block-bootstrap power and calibration, conformal quantile and coverage, rolling
  conformal uses only residuals observed at forecast time
* **data integrity** - no NaN/negatives, daily consecutive dates, totals match JHU scale, connected graph

GitHub Actions (`.github/workflows/`):
* `ci.yml` - on every push/PR: ruff lint -> tests on Python 3.10/3.11/3.12 (CPU-only PyTorch) with coverage artifacts
  -> 2-epoch smoke training run
* `full-pipeline.yml` - manual button: downloads the JHU data, re-runs the complete benchmark and uploads results/figures

## Extensions beyond the base model
* `qepignn_hybrid` - unitary + heat-kernel branches (exploratory, post-hoc; did **not** improve results)
* `src/stats.py` - paired moving-block bootstrap of RMSE differences (-> results/significance.json)
* `src/uncertainty.py` - split-conformal 90% intervals, static and rolling (-> results/uncertainty.json)
* `demo/qepignn_demo.html` - self-contained Forecast Explorer (open in a browser; rebuild with `python demo/build_demo.py`)
* `Dockerfile` - `docker build -t qepignn . && docker run --rm qepignn` runs the tests in a pinned CPU environment

## Headline findings (honest summary)
Persistence is hard to beat at 7-14 days in the low-incidence test period. Q-EpiGNN's 30-day gain over GCN-LSTM is
not statistically significant. A classical heat-diffusion variant with the same amplitude encoding was best overall and
significantly better than the unitary quantum walk. Conformal intervals are near-nominal at 7-14 days, under-cover at 30.
