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

<img src="../figures/ed_spectrum.png" alt="ED spectrum N=9 and N=18" width="560">

---

## 02 — VQE Ground State

**File:** [`02_vqe_run.ipynb`](../notebooks/02_vqe_run.ipynb)  
**What it does:**
- **COBYLA** (primary): 5 seeds × 5000 evaluations via `run_vqe_cobyla_multi_seed`, HEA depth=3
- **Adam** (diagnostic): 2 seeds × 1000 steps — demonstrates the zero-gradient failure
- Statistical summary: mean ± std, min/max across seeds; per-seed CSV
- Fan plot of all seed convergence curves + box plot of final energies vs ED
- Saves best statevector to `data/statevector_hea_best.npy` for NB03

**Why COBYLA, not Adam:** `|0⟩⊗N` is a Z-basis eigenstate. All IsingXX/YY/ZZ gradients cancel
by SU(2) symmetry → Adam has nothing to follow. COBYLA samples the energy landscape
directly without needing gradients.

**Key outputs:**
- `figures/vqe_convergence.png`
- `figures/vqe_bar.png`
- `figures/vqe_seed_distribution.png`
- `data/vqe_results.csv` (includes `mean_energy`, `std_energy`, `n_seeds`, `min_energy`, `max_energy`)
- `data/vqe_seeds_n9.csv`
- `data/statevector_hea_best.npy`

**Results:**

| N | Seeds | Mean E₀ | Std E₀ | Best E₀ | Error (best) |
|---|-------|---------|--------|---------|--------------|
| 9 | 5 | −1.23572 | 0.02853 | −1.28456 | **9.66%** |

| Method | E₀ (best) | Error vs ED | Evals |
|--------|-----------|-------------|-------|
| COBYLA / HEA depth=3 | −1.28456 | **9.66%** | 801 |
| Adam / HEA depth=3 | +0.141 | stalled | 1000 |
| ED exact | −1.42190399 | — | — |

<img src="../figures/vqe_bar.png" alt="VQE vs ED bar chart" width="420">
<img src="../figures/vqe_seed_distribution.png" alt="COBYLA seed distribution N=9" width="420">

---

## 03 — Entanglement Analysis

**File:** [`03_entanglement.ipynb`](../notebooks/03_entanglement.ipynb)  
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

<table>
<tr>
<td><img src="../figures/entanglement_sublattice_mi.png" alt="Sublattice MI" width="100%"></td>
<td><img src="../figures/entanglement_mi_matrix.png" alt="Pairwise MI matrix" width="100%"></td>
</tr>
</table>

---

## 04 — SOC QAOA

**File:** [`04_soc_qaoa.ipynb`](../notebooks/04_soc_qaoa.ipynb)  
**What it does:**
- Loads θ_SH dataset from `data/mp_theta_sh.csv` (12 spintronic materials; MP descriptors + literature θ_SH)
- Trains MLP surrogate (`surrogate.train_surrogate`)
- Scatter plot: actual vs predicted θ_SH
- Formulates k=3 from N=12 selection as QUBO with constraint penalty λ=6
- Runs QAOA at depth p=1, 2, 3 (COBYLA, 5 seeds × 300 evals per depth)
- **p=1 (γ, β) cost landscape** with COBYLA path; right panel plots total θ_SH vs depth
  (values pinned to `data/qaoa_results.csv` so the diagnostic matches published rankings)
- Classical baselines: greedy top-k and simulated annealing
- Bar comparison and material ranking visualisation

**Key outputs:**
- `figures/surrogate_predictions.png`
- `figures/qaoa_comparison.png`
- `figures/qaoa_convergence.png`
- `figures/qaoa_landscape.png`
- `figures/qaoa_material_ranking.png`
- `data/qaoa_results.csv`

### Data provenance and scope boundary

`data/mp_theta_sh.csv` combines two different kinds of information:

- Materials Project metadata: `mp_id`, formula, crystal system, space group, band gap, magnetic flag, and related structure descriptors.
- Curated literature labels: `theta_sh` and anomalous Hall conductivity values used as the supervised target for the surrogate.

Materials Project is therefore used as a structure and metadata source here; it is not treated as the source of the spin Hall angle labels. The refresh script (`scripts/fetch_mp_theta_sh.py`) can update Materials Project descriptors when `MP_API_KEY` is available, while the `theta_sh` labels remain curated literature inputs unless explicitly revised.

NB04 should be read as a small reproducibility and method-demonstration notebook:

- It tests whether the committed 12-material table can drive a surrogate-plus-QAOA workflow end to end.
- It compares QAOA selections against greedy and simulated-annealing baselines on the same oracle.
- It does not claim a new experimentally validated material discovery.
- It does not replace a larger literature review, DFT campaign, or laboratory validation.

The main guarded conclusion is that, for the current 12-material surrogate oracle, classical baselines outperform the tested QAOA depths. That is a useful negative/diagnostic result and a boundary for future scaling work.

**Results (k=3 from N=12 materials):**

| Method | Total θ_SH | Selected |
|--------|-----------|----------|
| QAOA p=1 | 3.049 | W, Ta, Bi₂Se₃ (sub-optimal) |
| QAOA p=2 | 3.049 | W, Ta, Bi₂Se₃ |
| QAOA p=3 | −0.451 | W, Ta, Pd |
| **Greedy** | **4.259** | **Bi₂Se₃, CrTe₂, Mn₃Sn** |
| Sim. annealing | 4.259 | Mn₃Sn, CrTe₂, Bi₂Se₃ |

With MP-grounded structure descriptors (`data/mp_theta_sh.csv`), classical
baselines beat QAOA on the surrogate oracle. QAOA p=1/2 tie as the best
quantum run; p=3 does not improve the selection on this 12-material problem.

<table>
<tr>
<td><img src="../figures/qaoa_material_ranking.png" alt="QAOA material ranking" width="100%"></td>
<td><img src="../figures/qaoa_comparison.png" alt="QAOA vs baselines" width="100%"></td>
</tr>
<tr>
<td colspan="2"><img src="../figures/qaoa_landscape.png" alt="QAOA p=1 landscape" width="100%"></td>
</tr>
</table>

---

## 05 — Scaling Analysis

**File:** [`05_scaling_analysis.ipynb`](../notebooks/05_scaling_analysis.ipynb)  
**What it does:**
- Sparse ED for N=12 inline (4096-dim Hilbert space, ~seconds)
- Loads N=9 COBYLA statistics from NB02 CSV (`mean_energy`, `vqe_seeds_n9.csv`)
- Runs COBYLA VQE at N=12 via `run_vqe_cobyla_multi_seed` (HEA depth=2, 3 seeds × 2000 evals)
- VQE energy error vs N (9, 12) + ED reference at N=18
- Box plot comparing seed energy distributions at N=9 vs N=12
- Adam gradient variance at N=9 and N=12 (10 seeds × 30 steps)
- Box plot + log-scale plot of barren plateau scaling
- Saves `data/vqe_scaling.csv` (includes seed statistics columns)

**Key outputs:**
- `figures/scaling_energy.png`
- `figures/scaling_gradient_variance.png`
- `figures/vqe_seed_scaling_boxplot.png`
- `data/vqe_scaling.csv`

**Results (COBYLA, after notebook rerun):**

| N | Seeds | Best E₀ | Error (best) | Notes |
|---|-------|---------|--------------|-------|
| 9 | 5 | −1.28456 | 9.66% | from NB02 CSV |
| 12 | 3 | −1.23859 | 16.33% | `run_vqe_cobyla_multi_seed`, seeds [42, 7, 123] |

<table>
<tr>
<td><img src="../figures/scaling_energy.png" alt="VQE error and energy vs N" width="100%"></td>
<td><img src="../figures/scaling_gradient_variance.png" alt="Gradient variance scaling" width="100%"></td>
</tr>
</table>

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

## NB04 provenance and scope

NB04 operates on **k=3 from N=12** using predictions from an in-sample surrogate fitted to the committed illustrative oracle. The near-diagonal surrogate scatter is therefore a training diagnostic, not an out-of-sample validation result. The ranking figure highlights exactly three entries because `k=3` is imposed by the QUBO constraint.

The committed targets are not asserted to be row-wise literature measurements. Consult [the provenance contract](../data/theta_sh_sources.md) and [machine-readable ledger](../data/theta_sh_provenance.csv). Results are limited to this fixed oracle and do not establish materials discovery, DFT-computed theta_SH, a global physical optimum, or quantum advantage.
