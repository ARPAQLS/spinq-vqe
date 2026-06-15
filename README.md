<div align="center">

<img src="docs/qondra_spin_vqe_logo.png" alt="QONDRA Spin VQE" width="500">

**Variational Quantum Simulation of Antiferromagnetic Hamiltonians**

Part of [ARPA Quantum Logical Systems — QONDRA](https://github.com/arpaqls) &nbsp;·&nbsp; [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

![Python](https://img.shields.io/badge/Python-3.11%2B-C7E4CA?style=flat-square&labelColor=756F6A)
![PennyLane](https://img.shields.io/badge/PennyLane-0.39%2B-DBD3DC?style=flat-square&labelColor=756F6A)
[![License](https://img.shields.io/badge/License-MIT-F4ECC8?style=flat-square&labelColor=756F6A)](LICENSE)
![SciPy](https://img.shields.io/badge/Optimizer-COBYLA%2FAdam-F0D9CC?style=flat-square&labelColor=756F6A)
![Status](https://img.shields.io/badge/Status-Research-EBD8DC?style=flat-square&labelColor=756F6A)

![A1](https://img.shields.io/badge/A1-VQE_Kagome_AFM-C7E4CA?style=flat-square&labelColor=756F6A)
![B2](https://img.shields.io/badge/B2-SOC_QAOA-DBD3DC?style=flat-square&labelColor=756F6A)
![Notebooks](https://img.shields.io/badge/Notebooks-3%2F5_complete-F0D9CC?style=flat-square&labelColor=756F6A)
![ED Error](https://img.shields.io/badge/VQE_error-9.66%25-EBD8DC?style=flat-square&labelColor=756F6A)

</div>

---

## What this is

`spinq-vqe` simulates the quantum many-body physics of **Mn₃Sn** — a Kagome antiferromagnet that demonstrated 40-picosecond spin-orbit torque switching (UTokyo, 2026). We use Variational Quantum Eigensolvers (VQE) to approximate its ground state and compare directly to spectroscopic data.

Two workstreams:

- **A1 — VQE on the Kagome Heisenberg AFM**: ground-state energy, entanglement structure, barren plateau diagnostics, exact diagonalization benchmarks.
- **B2 — SOC QAOA**: classical MLP surrogate on Materials Project spin Hall angle data, used as oracle for a QAOA composition optimizer.

## Structure

```
spinq-vqe/
├── src/spinq_vqe/
│   ├── kagome.py        # Kagome lattice graph + Heisenberg Hamiltonian
│   ├── ansatz.py        # HVA, HEA, MERA variational ansatze
│   ├── vqe.py           # COBYLA (primary) + Adam (diagnostic) VQE runners
│   ├── entanglement.py  # Von Neumann entropy, mutual information
│   ├── utils.py         # Publication-quality plot helpers
│   ├── surrogate.py     # MLP surrogate on MP θ_SH data  [B2]
│   └── qaoa.py          # QAOA circuit + optimizer        [B2]
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
Optional: `pip install -e ".[data]"` adds `scikit-learn`, `mp-api`, `pandas` (for B2 workstream).

## Notebooks

| # | Notebook | Status |
|---|----------|--------|
| 01 | [`01_kagome_hamiltonian.ipynb`](notebooks/01_kagome_hamiltonian.ipynb) | ✅ Complete — lattice, ED baseline, figures |
| 02 | [`02_vqe_run.ipynb`](notebooks/02_vqe_run.ipynb) | ✅ Complete — COBYLA 9.66% error, Adam barren plateau confirmed |
| 03 | [`03_entanglement.ipynb`](notebooks/03_entanglement.ipynb) | ✅ Complete — entropy profile, MI matrix, sublattice correlations |
| 04 | `04_soc_qaoa.ipynb` | 🔲 Not started |
| 05 | `05_scaling_analysis.ipynb` | 🔲 Not started |

## Key results

### A1 — Ground-state energy (N=9 Kagome Heisenberg AFM)

| Method | E₀ (normalized) | Error vs ED | Notes |
|--------|-----------------|-------------|-------|
| Exact diag. (N=9) | −1.42190399 | — | Sparse ED, gap Δ ≈ 0 (degenerate) |
| Exact diag. (N=18) | −1.49962859 | — | Spectral gap Δ = 0.037 |
| **COBYLA / HEA depth=3** | **−1.28456** | **9.66%** | 27 params, 801 evaluations |
| Adam / HEA depth=3 | +0.141 | — | Stalled at ferromagnetic saddle (barren plateau) |

**Why COBYLA, not Adam:** The `|0⟩⊗N` initial state is a Z-basis eigenstate — all IsingXX/YY/ZZ gradients cancel to exactly zero by SU(2) symmetry. Adam has no signal to follow. COBYLA uses function evaluations directly and is immune to this.

### A1 — Entanglement structure (COBYLA VQE statevector)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean single-site entropy | **0.9066 bits** | Near-maximal → strong quantum fluctuations |
| Max single-site entropy | **1.000 bits** | 7 of 9 sites maximally entangled |
| Sublattice I(A:B) | **3.689 bits** | Strong inter-sublattice correlations |
| Sublattice I(A:C), I(B:C) | **2.235 bits** | C sublattice also correlated |
| Mean pairwise MI | **0.227 bits** | Non-local correlations (spin liquid signature) |

## Docs

→ [`docs/`](docs/README.md) — physics background, ansatz guide, API reference, notebook guide.

## References

See [`REFERENCES.md`](REFERENCES.md) for the full bibliography.  
Key: Sachdev (1992), Yan/Huse/White (2011), Wiersema et al. (2020), Kandala et al. (2017), Cerezo et al. (2021), Farhi et al. (2014).

---

**License:** MIT &nbsp;·&nbsp; **Contact:** [qondra@arpacorp.net](mailto:qondra@arpacorp.net)
