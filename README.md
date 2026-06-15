# spinq-vqe

**Variational Quantum Simulation of Antiferromagnetic Hamiltonians**

> Part of [ARPA Quantum Logical Systems — QONDRA](https://github.com/arpaqls) &nbsp;·&nbsp; [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

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
│   ├── ansatz.py        # HVA (primary), HEA, MERA ansatze
│   ├── vqe.py           # VQE runner + Adam optimizer
│   ├── entanglement.py  # Von Neumann entropy, mutual information
│   ├── utils.py         # Publication-quality plot helpers
│   ├── surrogate.py     # MLP surrogate on MP θ_SH data  [WIP]
│   └── qaoa.py          # QAOA circuit + optimizer        [WIP]
├── notebooks/           # Executable research notebooks
├── figures/             # Generated plots
├── data/                # ED reference energies, VQE results
├── docs/                # Guides and API reference → docs/index.md
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

Requires Python ≥ 3.11. Core dependencies: `pennylane ≥ 0.38`, `numpy`, `scipy`, `networkx`, `matplotlib`.

## Notebooks

| # | Notebook | Status |
|---|----------|--------|
| 01 | [`01_kagome_hamiltonian.ipynb`](notebooks/01_kagome_hamiltonian.ipynb) | ✅ Complete — lattice, ED baseline, figures |
| 02 | [`02_vqe_run.ipynb`](notebooks/02_vqe_run.ipynb) | 🔬 Running — HVA primary, HEA comparison |
| 03 | [`03_entanglement.ipynb`](notebooks/03_entanglement.ipynb) | ⏳ Pending NB02 |
| 04 | `04_soc_qaoa.ipynb` | 🔲 Not started |
| 05 | `05_scaling_analysis.ipynb` | 🔲 Not started |

## Key results so far

| System | Method | E₀ (normalized) | Notes |
|--------|--------|-----------------|-------|
| N=9 Kagome | Exact diag. | −1.42190399 | Sparse ED, Δ ≈ 0 (degenerate) |
| N=18 Kagome | Exact diag. | −1.49962859 | Spectral gap Δ = 0.037 |
| N=9 | HVA (depth=6) | in progress | Target: < 5% error |

## Docs

→ [`docs/`](docs/README.md) — physics background, ansatz guide, API reference, notebook guide.

## References

See [`REFERENCES.md`](REFERENCES.md) for the full bibliography.  
Key: Sachdev (1992), Yan/Huse/White (2011), Wiersema et al. (2020), Kandala et al. (2017), Cerezo et al. (2021).

---

**License:** MIT &nbsp;·&nbsp; **Contact:** [qondra@arpacorp.net](mailto:qondra@arpacorp.net)
