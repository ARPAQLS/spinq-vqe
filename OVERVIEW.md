# spinq-vqe — Research Overview

**Variational Quantum Simulation of Antiferromagnetic Hamiltonians**

Part of [ARPA Quantum Logical Systems — QONDRA](https://github.com/arpaqls) · [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

---

## What this is

In May 2026, the University of Tokyo (Nakatsuji Lab) demonstrated **40-picosecond spin-orbit torque switching** in **Mn₃Sn / tantalum heterostructures** — a Kagome antiferromagnet whose switching is rooted in spin–orbit coupling (SOC).

We cannot fabricate this material in silico, but we *can* simulate its quantum many-body physics using **Variational Quantum Eigensolvers (VQE)** and **Quantum Approximate Optimization Algorithms (QAOA)** on a classical simulator — and compare results directly to exact diagonalization and spectroscopic benchmarks.

`spinq-vqe` is the open-source Python package that implements this pipeline: lattice construction, variational ansätze, VQE runners, entanglement analysis, an MLP surrogate for spin Hall angle, and a QAOA material-selection optimizer. All five research notebooks are executed with published figures and reference data.

---

## Research threads

### Kagome lattice VQE

We simulate the Heisenberg antiferromagnet on a **1D Kagome strip** (the geometry used in the notebooks):

```
H = J Σ_{<i,j>} S_i · S_j  +  D Σ_i (S_i^z)²  +  B Σ_i S_i^z
```

The pipeline builds the lattice graph (NetworkX), maps spin operators to Pauli strings (PennyLane), and runs VQE with **COBYLA** (primary) or **Adam** (diagnostic). Results are benchmarked against sparse exact diagonalization.

**Decisions made in this release:** HEA depth = 3 for N = 9; COBYLA over Adam (zero gradients from the `|0⟩⊗N` initial state); system sizes N = 9, 12, and 18 explored in the scaling notebook.

### SOC material screening via QAOA

The spin Hall angle (θ_SH) is the figure of merit for spin-orbit torque efficiency. Selecting the top-k materials from N candidates is a combinatorial optimization problem, solved here with **QAOA** using a classical **MLP surrogate** as the oracle (trained on a 12-material mock dataset; Materials Project API supported via `load_mp_data()`).

**Decisions made in this release:** classical surrogate oracle (not raw DFT per evaluation); QAOA depths p = 1, 2, 3 compared against greedy and simulated-annealing baselines; k = 3 selected from N = 12 materials.

---

## Key results at a glance

### Ground-state energy (VQE vs exact diagonalization)

| N | Method | E₀ (normalized) | Error vs ED | Notes |
|---|--------|-----------------|-------------|-------|
| 9  | Exact diag. | −1.42190399 | — | Sparse ED, gap Δ ≈ 0 |
| 9  | **COBYLA / HEA d=3** | **−1.28456** | **9.66%** | 27 params, 801 evals |
| 9  | Adam / HEA d=3 | +0.141 | — | Barren plateau stall |
| 12 | COBYLA / HEA d=2 | −1.20351 | 18.70% | 24 params |
| 18 | Exact diag. | −1.49962859 | — | Spectral gap Δ = 0.037 |

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
| QAOA p=1 | 3.080 | W, Ta, Bi₂Se₃ | Shallow circuit — sub-optimal |
| **QAOA p=2** | **4.263** | **Mn₃Sn, CrTe₂, Bi₂Se₃** | Matches global optimum |
| QAOA p=3 | 4.263 | Mn₃Sn, CrTe₂, Bi₂Se₃ | Confirms p=2 result |
| Greedy (classical) | 4.263 | Bi₂Se₃, CrTe₂, Mn₃Sn | Optimal baseline |

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
| Notebooks | `notebooks/01`–`05` | Lattice/ED, VQE, entanglement, SOC QAOA, scaling |
| Test suite | `tests/` | Six modules, < 90 s on CPU |
| Documentation | `docs/` | Physics, ansätze, API, notebooks, testing |
| Data & figures | `data/`, `figures/` | ED energies, VQE/QAOA CSVs, publication plots |

---

## What's next

- **DMRG comparison** — benchmark VQE against density-matrix renormalization group at larger system sizes
- **Real Materials Project data** — replace mock surrogate training set with live `mp-api` fetches
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

> spinq-vqe: Variational Quantum Simulation of Antiferromagnetic Hamiltonians. ARPA Quantum Logical Systems (QONDRA). https://github.com/ARPAQLS/spinq-vqe

A Zenodo DOI will be added when the first paper-linked release is archived.

---

*Last updated: 2026-07-03 · Part of the ARPA Spintronics QML Research Program*
