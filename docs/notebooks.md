# Notebook Guide

> [← index](index.md)

---

## Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
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
| 9 | −1.42190399 | ≈ 0 |
| 18 | −1.49962859 | 0.037 |

---

## 02 — VQE Ground State: HVA vs HEA

**File:** [`02_vqe_run.ipynb`](../notebooks/02_vqe_run.ipynb)  
**Status:** 🔬 Running

**What it does:**
- Runs VQE with 5 random seeds for both HVA (primary) and HEA (comparison)
- HVA: depth=6, 18 params, scale=0.05 init — no barren plateau
- HEA: depth=3, 27 params, scale=1.0 init — used as baseline
- Plots convergence curves and gradient variance (barren plateau diagnostic)
- Saves best statevectors for Notebook 03

**Key outputs:**
- `figures/vqe_convergence.png`
- `figures/vqe_bar.png`
- `data/vqe_results.csv`
- `data/statevector_hva_best.npy`
- `data/statevector_hea_best.npy`

**Expected results:** HVA < 10% error from ED; HEA > 50% error (barren plateau demonstration).

---

## 03 — Entanglement Analysis

**File:** [`03_entanglement.ipynb`](../notebooks/03_entanglement.ipynb)  
**Status:** ⏳ Pending NB02

**Depends on:** `data/statevector_hva_best.npy`, `data/statevector_hea_best.npy`

**What it will do:**
- Von Neumann entropy bipartition scan (all contiguous subsystems of size 1 → 8)
- 9×9 pairwise mutual information matrix
- 3×3 sublattice mutual information matrix (A↔B↔C)
- Single-site entropies across all 9 sites
- Compare entanglement structure of HVA vs HEA statevectors

**Key outputs:** `figures/entanglement_*.png`

---

## 04 — SOC QAOA *(not started)*

**File:** `04_soc_qaoa.ipynb`

**What it will do:**
- Train MLP surrogate on Materials Project θ_SH data
- Formulate SOC optimization as QUBO
- Run QAOA at depth p=1..5 vs classical simulated annealing
- Rank candidate heterostructure compositions by predicted spin Hall angle

---

## 05 — Scaling Analysis *(not started)*

**File:** `05_scaling_analysis.ipynb`

**What it will do:**
- VQE energy error vs system size (N=9, 18, 27)
- Circuit depth required for < 5% error at each N
- Gradient variance scaling with N (barren plateau characterization)
- Estimate crossover point where VQE breaks down vs DMRG

---

## Running a specific notebook

```bash
# Execute in-place (saves outputs to the .ipynb file)
.venv\Scripts\python.exe -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=spinq-vqe \
  notebooks/01_kagome_hamiltonian.ipynb
```

Or simply open JupyterLab and run interactively.
