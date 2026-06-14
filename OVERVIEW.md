# spinq-vqe
## Variational Quantum Simulation of Antiferromagnetic Hamiltonians

> **Clusters:** A1 (AFM Kagome VQE) + B2 (SOC QAOA)
> **Status:** Planning
> **Priority:** 🥇 High — establishes the physics foundation all other repos reference

---

## Motivation

The University of Tokyo team (Nakatsuji Lab, May 2026) demonstrated 40-picosecond switching in
**Mn₃Sn / tantalum heterostructures** — a Kagome antiferromagnet. The switching is driven by
spin-orbit torque (SOT), a phenomenon rooted in spin-orbit coupling (SOC).

We cannot fabricate this material. But we *can* simulate its quantum many-body physics using
Variational Quantum Eigensolvers (VQE) and Quantum Approximate Optimization Algorithms (QAOA)
on a classical simulator — and publish results that are directly comparable to the lab's
spectroscopic data.

This repo puts QML tools directly in contact with the physics that is dominating the
experimental spintronics literature.

---

## Core Workstreams

### Workstream A1 — VQE for Antiferromagnetic Kagome Lattice
Simulate the **Heisenberg antiferromagnet on a Kagome lattice** (the geometry of Mn₃Sn):

```
H = J Σ_{<i,j>} S_i · S_j  +  D Σ_i (S_i^z)²  +  B Σ_i S_i^z
```

Where:
- `J > 0` → antiferromagnetic exchange
- `D` → single-ion anisotropy (relevant for Mn₃Sn)
- `B` → external field

Tasks:
- Build the Kagome lattice graph (NetworkX)
- Map spin operators to Pauli strings (PennyLane / OpenFermion)
- Implement VQE with hardware-efficient ansatz and UCCSD variants
- Benchmark ground-state energy vs exact diagonalization (QuSpin / NumPy)
- Analyze entanglement entropy across lattice sizes (8 → 24 sites)
- Study frustration: the Kagome lattice has no classical Néel ground state — quantum
  fluctuations dominate, which is exactly why VQE should outperform mean-field

### Workstream B2 — QAOA for Spin-Orbit Coupling Optimization
The spin Hall angle (θ_SH) is the figure-of-merit for SOT efficiency. Maximizing it over
a combinatorial space of heterostructure compositions is a **QUBO problem**:

```
Maximize:  θ_SH(x)  subject to  Σ x_i = k  (choose k layers from N candidates)
```

Tasks:
- Formulate the SOC optimization as a QUBO using published DFT data as a surrogate oracle
- Encode on QAOA circuit (PennyLane `qml.qaoa` module)
- Compare QAOA depth p=1..5 vs classical simulated annealing
- Use Materials Project API to get real θ_SH estimates for candidate materials
- Visualize the optimization landscape (QAOA energy landscape plots)

---

## Stack

| Component | Library | Role |
|-----------|---------|------|
| Primary QML | PennyLane ≥ 0.38 | VQE, QAOA, circuit ops |
| Autodiff / GPU | JAX + jaxlib | Fast gradient computation |
| Spin physics | QuSpin | Exact diagonalization baseline |
| Lattice geometry | NetworkX | Kagome graph construction |
| Materials data | `mp-api` (Materials Project) | Real SOC / θ_SH data |
| Hamiltonian tools | OpenFermion | Pauli string mapping |
| Numerics | NumPy, SciPy | Support |
| Visualization | Matplotlib, Plotly | Publication-quality figures |
| Notebooks | Jupyter | Reproducible demos |

Python ≥ 3.11 throughout. No Rust/C required.

---

## Deliverables

- [ ] `src/spinq_vqe/` — Python package with Kagome Hamiltonian builder, VQE runner, QAOA optimizer
- [ ] `notebooks/01_kagome_hamiltonian.ipynb` — Lattice construction, Pauli mapping walkthrough
- [ ] `notebooks/02_vqe_ground_state.ipynb` — VQE convergence, energy vs exact diag
- [ ] `notebooks/03_entanglement_entropy.ipynb` — Quantum correlations across frustration regimes
- [ ] `notebooks/04_soc_qaoa.ipynb` — QUBO formulation, QAOA optimization, material ranking
- [ ] `notebooks/05_scaling_analysis.ipynb` — NISQ scaling limits, circuit depth vs accuracy
- [ ] `data/` — Pre-computed exact diag results, Materials Project SOC data (cached)
- [ ] `paper/` — LaTeX draft targeting Physical Review B or npj Quantum Materials

---

## Paper Angle

**Title (draft):** *Variational Quantum Simulation of Frustrated Antiferromagnets and
Spin-Orbit Coupling Optimization for Spintronic Device Design*

**Novelty:**
1. First VQE study explicitly targeting the Mn₃Sn Kagome Hamiltonian with comparison
   to published inelastic neutron scattering data
2. QAOA reframing of the SOC engineering problem — connects quantum optimization
   to experimental materials design
3. Systematic NISQ scaling analysis: at what system size does VQE break down vs DMRG?

**Target Journals:** Physical Review B, npj Quantum Materials, Quantum Science and Technology

**arXiv sections:** `cond-mat.str-el`, `quant-ph`

---

## Unique Selling Point

> This is the only repo in the set that directly simulates the **material physics**
> behind the UTokyo experiment. Every other repo builds on this physical foundation.
> The Kagome AFM Hamiltonian computed here is used as the reservoir Hamiltonian in
> `spintronic-qrc` and as the noise-source model in `mtj-quantum-noise`.

---

## Cross-Repo Dependencies

```
spinq-vqe (this repo)
    │
    ├──► spintronic-qrc         [provides: Kagome Hamiltonian as QRC reservoir]
    ├──► mtj-quantum-noise      [provides: AFM spin dynamics as decoherence source]
    └──► quantum-hopfield-mram  [provides: Ising-limit Hamiltonian for memory landscape]
```

---

## Research Context & Key References

1. **Nakatsuji et al. (2026)** — UTokyo Mn₃Sn SOT switching, 40 ps timescale
2. **Sachdev (1992)** — Kagome Heisenberg antiferromagnet, spin liquid phases
3. **Peruzzo et al. (2014)** — Original VQE paper (Nature Communications)
4. **Farhi et al. (2014)** — Original QAOA paper
5. **Sinova et al. (2015)** — Spin Hall effects in materials (Rev. Mod. Phys.)
6. **Yan et al. (2011)** — Spin liquid in Kagome Heisenberg (Science)

---

## Getting Started (when ready)

```bash
# Create environment
conda create -n spinq-vqe python=3.11
conda activate spinq-vqe

# Install core deps
pip install pennylane pennylane-lightning[gpu] jax jaxlib quspin openfermion
pip install networkx mp-api matplotlib plotly jupyter

# Run first notebook
jupyter lab notebooks/01_kagome_hamiltonian.ipynb
```

---

## Open Questions / Design Decisions

- [ ] Ansatz choice for Kagome VQE: hardware-efficient vs physics-inspired (UCCSD, MERA)?
- [ ] System sizes: start at 9 sites (1 unit cell × 3 sublattices) → scale to 18, 27?
- [ ] QAOA oracle: use DFT-computed θ_SH from literature or train a classical surrogate first?
- [ ] Should QUBO include thermal stability constraints (energy barrier > 40 kT)?
- [ ] GPU backend: PennyLane-Lightning-GPU or JAX-based custom simulation?

---

*Last updated: 2026-05-24 | Part of the ARPA Spintronics QML Research Program*
