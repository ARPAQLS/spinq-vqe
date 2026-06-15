# Notebook Guide

> [← index](README.md)

---

## Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -e ".[dev]"
jupyter lab
```

All notebooks live in `notebooks/`. Run them in order — later notebooks depend on outputs from earlier ones.

---

## 01 — Kagome Hamiltonian & Exact Diagonalization

**File:** [`01_kagome_hamiltonian.ipynb`](../notebooks/01_kagome_hamiltonian.ipynb)  
**Status:** ✅ Complete

**What it does:**
- Builds the 9-site and 18-site Kagome lattice graphs
- Constructs the Heisenberg + anisotropy Hamiltonian as a PennyLane `Hamiltonian`
- Plots the lattice with sublattice coloring and Hamiltonian coefficient distribution
- Runs sparse exact diagonalization, extracts ground state energy and spectral gap
- Saves `data/ed_reference_energies.csv` (used by NB02)

**Key outputs:**
- `figures/kagome_lattice.png`
- `figures/hamiltonian_coeffs.png`
- `figures/ed_spectrum.png`
- `data/ed_reference_energies.csv`

**Validated results:**

| N | E₀ (normalized) | Gap Δ |
|---|-----------------|-------|
| 9 | −1.42190399 | ≈ 0 (degenerate) |
| 18 | −1.49962859 | 0.037 |

---

## 02 — VQE Ground State

**File:** [`02_vqe_run.ipynb`](../notebooks/02_vqe_run.ipynb)  
**Status:** ✅ Complete

**What it does:**
- **COBYLA** (primary): 5 seeds × 5000 evaluations, HEA depth=3, random init `scale=1.0`
- **Adam** (diagnostic): 2 seeds × 1000 steps — demonstrates the zero-gradient failure
- Plots COBYLA convergence curves + Adam gradient variance (barren plateau evidence)
- Saves best statevector to `data/statevector_hea_best.npy` for NB03

**Why COBYLA, not Adam:** `|0⟩⊗N` is a Z-basis eigenstate. All IsingXX/YY/ZZ gradients cancel
by SU(2) symmetry → Adam has nothing to follow. COBYLA samples the energy landscape
directly without needing gradients.

**Key outputs:**
- `figures/vqe_convergence.png`
- `figures/vqe_bar.png`
- `data/vqe_results.csv`
- `data/statevector_hea_best.npy`

**Results:**

| Method | E₀ | Error vs ED | Evals |
|--------|----|-------------|-------|
| COBYLA / HEA depth=3 | −1.28456 | **9.66%** | 801 |
| Adam / HEA depth=3 | +0.141 | stalled | 1000 |
| ED exact | −1.42190399 | — | — |

---

## 03 — Entanglement Analysis

**File:** [`03_entanglement.ipynb`](../notebooks/03_entanglement.ipynb)  
**Status:** ✅ Complete

**Depends on:** `data/statevector_hea_best.npy`

**What it does:**
- Single-site Von Neumann entropy for all 9 sites
- Bipartition scan: S vs subsystem size |A| = 1 → 4
- 9×9 pairwise mutual information matrix
- 3×3 sublattice mutual information matrix (A↔B, A↔C, B↔C)
- Spin liquid diagnostic interpretation

**Key outputs:**
- `figures/entanglement_bipartition.png`
- `figures/entanglement_mi_matrix.png`
- `figures/entanglement_sublattice_mi.png`

**Results:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean single-site S | 0.9066 bits | Near-maximal → strong fluctuations |
| Max single-site S | 1.000 bits | 7/9 sites maximally entangled |
| Sublattice I(A:B) | 3.689 bits | Strong inter-sublattice correlations |
| Sublattice I(A:C/B:C) | 2.235 bits | Full 3-way entanglement |
| Mean pairwise MI | 0.227 bits | Long-range non-local correlations |

Site 2 shows anomalously low entropy (0.235 bits) consistent with specific frustrated geometry.
Sites 0 and 1 form a near-perfect Bell pair (singlet on that bond).

---

## 04 — SOC QAOA *(not started)*

**File:** `04_soc_qaoa.ipynb`

**Depends on:** `surrogate.py`, `qaoa.py` (both implemented)

**What it will do:**
- Load mock θ_SH dataset (or query Materials Project API with key)
- Train MLP surrogate (`surrogate.train_surrogate`)
- Formulate k-from-N heterostructure selection as QUBO
- Run QAOA depth p=1..5 vs classical greedy + simulated annealing
- Rank candidate compositions by predicted spin Hall angle

---

## 05 — Scaling Analysis *(not started)*

**File:** `05_scaling_analysis.ipynb`

**What it will do:**
- VQE energy error vs system size (N=9, 18, 27)
- COBYLA evaluations required for < 10% error at each N
- Gradient variance scaling with N (Adam barren plateau characterization)
- Estimate crossover where COBYLA becomes infeasible vs DMRG

---

## Running a specific notebook

```bash
# Execute in-place (saves outputs to the .ipynb file)
.venv\Scripts\python.exe -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=spinq-vqe \
  notebooks/01_kagome_hamiltonian.ipynb
```

Or open JupyterLab and run interactively.
