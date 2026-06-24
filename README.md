<div align="center">

<img src="docs/qondra_spin_vqe_logo.png" alt="QONDRA Spin VQE" width="500">

**Variational Quantum Simulation of Antiferromagnetic Hamiltonians**

Part of [ARPA Quantum Logical Systems — QONDRA](https://github.com/arpaqls) &nbsp;·&nbsp; [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

<br>

![Python](https://img.shields.io/badge/Python-3.11%2B-C7E4CA?style=flat-square&labelColor=756F6A)
![PennyLane](https://img.shields.io/badge/PennyLane-0.39%2B-DBD3DC?style=flat-square&labelColor=756F6A)
[![License](https://img.shields.io/badge/License-MIT-F4ECC8?style=flat-square&labelColor=756F6A)](LICENSE)
![Optimizer](https://img.shields.io/badge/Optimizer-COBYLA%2FAdam-F0D9CC?style=flat-square&labelColor=756F6A)


</div>

---

## What this is

`spinq-vqe` simulates the quantum many-body physics of **Mn₃Sn** — a Kagome antiferromagnet that demonstrated 40-picosecond spin-orbit torque switching (UTokyo, 2026). We use Variational Quantum Eigensolvers (VQE) to approximate its ground state and compare directly to spectroscopic data.

Two parallel research threads:

- **VQE on the Kagome lattice**: ground-state energy, entanglement structure, barren plateau diagnostics, exact diagonalization benchmarks.
- **SOC material screening via QAOA**: classical MLP surrogate on spin Hall angle data, used as oracle for a QAOA composition optimizer.

## Structure

```
spinq-vqe/
├── src/spinq_vqe/
│   ├── kagome.py        # Kagome lattice graph + Heisenberg Hamiltonian
│   ├── ansatz.py        # HVA, HEA, MERA variational ansatze
│   ├── vqe.py           # COBYLA (primary) + Adam (diagnostic) VQE runners
│   ├── entanglement.py  # Von Neumann entropy, mutual information
│   ├── utils.py         # Publication-quality plot helpers
│   ├── surrogate.py     # MLP surrogate for spin Hall angle prediction
│   └── qaoa.py          # QAOA circuit + optimizer for material selection
├── notebooks/           # Executable research notebooks
├── figures/             # Generated plots
├── data/                # ED reference energies, VQE results, statevectors
├── docs/                # Guides and API reference → docs/README.md
├── OVERVIEW.md          # Full program description + research context
└── REFERENCES.md        # Full bibliography (50+ references)
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS
pip install -e ".[dev]"
```

Requires Python ≥ 3.11. Core: `pennylane ≥ 0.39`, `numpy`, `scipy`, `networkx`, `matplotlib`.  
Optional: `pip install -e ".[data]"` adds `scikit-learn`, `mp-api`, `pandas` (for SOC QAOA notebooks).

## Notebooks

| # | Notebook | Notes |
|---|----------|-------|
| 01 | [`01_kagome_hamiltonian.ipynb`](notebooks/01_kagome_hamiltonian.ipynb) | lattice, ED baseline, figures |
| 02 | [`02_vqe_run.ipynb`](notebooks/02_vqe_run.ipynb) | COBYLA 9.66% error, Adam barren plateau confirmed |
| 03 | [`03_entanglement.ipynb`](notebooks/03_entanglement.ipynb) | entropy profile, MI matrix, sublattice correlations |
| 04 | [`04_soc_qaoa.ipynb`](notebooks/04_soc_qaoa.ipynb) | surrogate MLP, QAOA p=1/2/3, material ranking |
| 05 | [`05_scaling_analysis.ipynb`](notebooks/05_scaling_analysis.ipynb) | N=9/12/18 scaling, gradient variance, barren plateau |

## Key results

### Ground-state energy (VQE vs exact diagonalisation)

| N | Method | E₀ (normalized) | Error vs ED | Notes |
|---|--------|-----------------|-------------|-------|
| 9  | Exact diag. | −1.42190399 | — | Sparse ED, gap Δ ≈ 0 |
| 9  | **COBYLA / HEA d=3** | **−1.28456** | **9.66%** | 27 params, 801 evals |
| 9  | Adam / HEA d=3 | +0.141 | — | Barren plateau stall |
| 12 | COBYLA / HEA d=2 | −1.20351 | 18.70% | 24 params |
| 18 | Exact diag. | −1.49962859 | — | Spectral gap Δ = 0.037 |

**Why COBYLA, not Adam:** The `|0⟩⊗N` initial state is a Z-basis eigenstate — all IsingXX/YY/ZZ gradients cancel to exactly zero by SU(2) symmetry. COBYLA uses function evaluations directly and is immune to this.

<table>
<tr>
<td><img src="figures/vqe_bar.png" alt="VQE vs ED" width="100%"></td>
<td><img src="figures/scaling_energy.png" alt="Scaling" width="100%"></td>
</tr>
</table>

### Entanglement structure (N=9 statevector)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean single-site entropy | **0.9066 bits** | Near-maximal → strong quantum fluctuations |
| Max single-site entropy | **1.000 bits** | 7 of 9 sites maximally entangled |
| Sublattice I(A:B) | **3.689 bits** | Strong inter-sublattice correlations |
| Sublattice I(A:C), I(B:C) | **2.235 bits** | C sublattice also correlated |
| Mean pairwise MI | **0.227 bits** | Non-local correlations (spin liquid signature) |

<img src="figures/entanglement_sublattice_mi.png" alt="Sublattice mutual information" width="420">

### SOC material selection via QAOA

| Method | Total θ_SH | Selected | Notes |
|--------|-----------|----------|-------|
| QAOA p=1 | 3.080 | W, Ta, Bi₂Se₃ | Shallow circuit — sub-optimal |
| **QAOA p=2** | **4.263** | **Mn₃Sn, CrTe₂, Bi₂Se₃** | Matches global optimum |
| QAOA p=3 | 4.263 | Mn₃Sn, CrTe₂, Bi₂Se₃ | Confirms p=2 result |
| Greedy (classical) | 4.263 | Bi₂Se₃, CrTe₂, Mn₃Sn | Optimal baseline |

<img src="figures/qaoa_material_ranking.png" alt="QAOA material ranking" width="560">

## Docs

→ [`docs/`](docs/README.md) — physics background, ansatz guide, API reference, notebook guide.

## References

See [`REFERENCES.md`](REFERENCES.md) for the full bibliography.  
Key: Sachdev (1992), Yan/Huse/White (2011), Wiersema et al. (2020), Kandala et al. (2017), Cerezo et al. (2021), Farhi et al. (2014).

---

**License:** MIT &nbsp;·&nbsp; **Contact:** [qondra@arpacorp.net](mailto:qondra@arpacorp.net)
