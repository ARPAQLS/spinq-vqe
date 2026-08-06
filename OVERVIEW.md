# spinq-vqe — Research Overview

**Variational Quantum Simulation of Antiferromagnetic Hamiltonians**

Part of [ARPA Quantum Logical Systems — QONDRA](https://github.com/arpaqls) · [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

---

## What this is

In May 2026, the University of Tokyo (Nakatsuji Lab) demonstrated **40-picosecond spin-orbit torque switching** in **Mn₃Sn / tantalum heterostructures** — a Kagome antiferromagnet whose switching is rooted in spin–orbit coupling (SOC).

We cannot fabricate this material in silico, but we *can* simulate its quantum many-body physics using **Variational Quantum Eigensolvers (VQE)** and **Quantum Approximate Optimization Algorithms (QAOA)** on a classical simulator — and compare results directly to exact diagonalization and spectroscopic benchmarks.

`spinq-vqe` is the open-source Python package that implements this pipeline: lattice construction, variational ansätze, VQE runners, entanglement analysis, TeNPy DMRG references, an MLP surrogate for spin Hall angle, and a QAOA material-selection optimizer. All six research notebooks are executed with published figures and reference data.

---

## Research threads

### Kagome lattice VQE

We simulate the Heisenberg antiferromagnet on a **1D Kagome strip** (the geometry used in the notebooks):

```
H = J Σ_{<i,j>} S_i · S_j  +  D Σ_i (S_i^z)²  +  B Σ_i S_i^z
```

The pipeline builds the lattice graph (NetworkX), maps spin operators to Pauli strings (PennyLane), and runs VQE with **COBYLA** (primary) or **Adam** (diagnostic). Results are benchmarked against sparse exact diagonalization.

**Decisions made in this release:** HEA depth = 3 for N = 9; COBYLA over Adam (zero gradients from the `|0⟩⊗N` initial state); system sizes N = 9, 12, 18, and 24 with DMRG as the primary classical reference beyond sparse ED.

### SOC material screening via QAOA

The spin Hall angle (θ_SH) is the figure of merit for spin-orbit torque efficiency. Selecting the top-k materials from N candidates is a combinatorial optimization problem, solved here with **QAOA** using a classical **MLP surrogate** as the oracle (trained on a 12-material dataset in `data/mp_theta_sh.csv`; optional refresh via Materials Project API).

Materials Project supplies the structure metadata, while the θ_SH targets are a fixed illustrative oracle (`illustrative_oracle` in `data/mp_theta_sh.csv`). NB04 is a reproducibility and method-demonstration workflow on the committed 12-material surrogate oracle rather than a new material-discovery claim.

**Decisions made in this release:** classical surrogate oracle (not raw DFT per evaluation); QAOA depths p = 1, 2, 3 compared against greedy and simulated-annealing baselines; k = 3 selected from N = 12 materials; p=1 (γ, β) landscape and depth-sensitivity plot (`figures/qaoa_landscape.png`) show that shallow QAOA can stall in a suboptimal basin while classical baselines reach the global optimum.

---

## Key results at a glance

### Ground-state energy (VQE vs exact diagonalization)

| N | Seeds | Mean E₀ | Std E₀ | Best E₀ | Error (best) | Notes |
|---|-------|---------|--------|---------|--------------|-------|
| 9 | 5 | −1.23572 | 0.02853 | **−1.28456** | **9.66%** | HEA d=3, 27 params |
| 12 | 3 | −1.21520 | 0.02026 | −1.23859 | 16.33% | HEA d=2, 24 params |
| 9 | — | — | — | −1.42190399 | — | Exact diag., gap Δ ≈ 0 |
| 18 | — | — | — | −1.49962859 | — | Exact diag., gap Δ = 0.037 |

Adam / HEA d=3 at N=9 stalls at +0.141 (barren plateau). Seed-level statistics live in
`data/vqe_results.csv` and `figures/vqe_seed_distribution.png`.

### Entanglement structure (N = 9 statevector)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean single-site entropy | **0.9066 bits** | Near-maximal → strong quantum fluctuations |
| Max single-site entropy | **1.000 bits** | 7 of 9 sites maximally entangled |
| Sublattice I(A:B) | **3.689 bits** | Strong inter-sublattice correlations |
| Mean pairwise MI | **0.227 bits** | Non-local correlations (spin liquid signature) |

### SOC material selection via QAOA

| Method | Total θ_SH | Selected | Notes |
|--------|-----------|----------|-------|
| QAOA p=1 | 3.049 | W, Ta, Bi₂Se₃ | Best QAOA depth — still sub-optimal |
| QAOA p=2 | 3.049 | W, Ta, Bi₂Se₃ | Same selection as p=1 |
| QAOA p=3 | −0.451 | W, Ta, Pd | Deeper circuit — worse on this oracle |
| **Greedy (classical)** | **4.259** | **Bi₂Se₃, CrTe₂, Mn₃Sn** | Optimal on surrogate oracle |
| Sim. annealing | 4.259 | Mn₃Sn, CrTe₂, Bi₂Se₃ | Matches greedy |

---

## Scientific context

This work sits at the intersection of frustrated magnetism, variational quantum algorithms, and spintronic materials design:

1. **Sachdev (1992)** — Kagome Heisenberg antiferromagnet and spin-liquid phases
2. **Yan, Huse & White (2011)** — spin liquid in the Kagome Heisenberg model
3. **Sinova et al. (2015)** — spin Hall effects in materials
4. **Peruzzo et al. (2014)** — original VQE
5. **Farhi et al. (2014)** — original QAOA
6. **Cerezo et al. (2021)** — barren plateaus in variational quantum algorithms

Full bibliography: [`REFERENCES.md`](REFERENCES.md) (50+ entries).

---

## What's in the repository

| Component | Location | Description |
|-----------|----------|-------------|
| Python package | `src/spinq_vqe/` | `kagome`, `ansatz`, `vqe`, `entanglement`, `surrogate`, `qaoa`, `utils` |
| Notebooks | `notebooks/01`–`06` | Lattice/ED, VQE, entanglement, SOC QAOA, scaling, DMRG |
| Test suite | `tests/` | Six modules, < 90 s on CPU |
| Documentation | `docs/` | Physics, ansätze, API, notebooks, testing |
| Data & figures | `data/`, `figures/` | ED energies, VQE/QAOA CSVs, publication plots |

---

## What's next

- **DMRG comparison** — benchmark VQE against density-matrix renormalization group at larger system sizes
- **Expand θ_SH dataset** — add materials or refresh descriptors via `scripts/fetch_mp_theta_sh.py`
- **2D periodic Kagome tiling** — extend beyond the 1D strip geometry
- **Paper draft** — LaTeX manuscript targeting Physical Review B or npj Quantum Materials

---

## Cross-repo dependencies

```
spinq-vqe (this repo)
    │
    ├──► spintronic-qrc         [Kagome Hamiltonian as QRC reservoir]
    ├──► mtj-quantum-noise      [AFM spin dynamics as decoherence source]
    └──► quantum-hopfield-mram  [Ising-limit Hamiltonian for memory landscape]
```

The Kagome AFM Hamiltonian computed here is intended as the physical foundation for related repos in the ARPA Spintronics QML program.

---

## Citation

If you use this software, please cite using the metadata in [`CITATION.cff`](CITATION.cff):

> Peilivanidis, V., & ARPA Quantum Logical Systems (QONDRA). (2026). *spinq-vqe: Variational Quantum Simulation of Antiferromagnetic Hamiltonians* (v0.1.5). https://doi.org/10.5281/zenodo.21628505

Zenodo concept DOI: [10.5281/zenodo.21628505](https://doi.org/10.5281/zenodo.21628505) (resolves to the latest archived version). Cite the software DOI for the code; cite any paper DOI separately when the manuscript is published.

---

*Last updated: 2026-08-06 · Part of the ARPA Spintronics QML Research Program*

## SOC QAOA provenance boundary

The SOC notebook is a workflow benchmark on a fixed illustrative oracle: 12 committed targets are fitted in-sample by the surrogate and used to select `k=3` candidates. The totals in the tables and figures are objective values on that oracle, not verified material constants. `data/theta_sh_sources.md` defines the status contract and `data/theta_sh_provenance.csv` binds every oracle row to either primary evidence or an explicit illustrative status. Replacing any target requires regenerating the surrogate, classical baselines, QAOA runs, tables, and figures together.
