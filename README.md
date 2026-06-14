# spinq-vqe

**Variational Quantum Simulation of Antiferromagnetic Hamiltonians**

> Part of [ARPA Quantum Logical Systems — QONDRA](https://github.com/arpaqls) &nbsp;·&nbsp; [qondra@arpacorp.net](mailto:qondra@arpacorp.net)

---

## What this is

`spinq-vqe` simulates the quantum many-body physics of **Mn₃Sn** — a Kagome antiferromagnet
that demonstrated 40-picosecond spin-orbit torque switching (UTokyo, 2026).
We cannot fabricate the material. We can simulate its ground-state structure using
Variational Quantum Eigensolvers (VQE) and compare directly to spectroscopic data.

The repo covers two workstreams:

- **A1 — VQE on the Kagome Heisenberg AFM**: hardware-efficient (HEA) and
  MERA-inspired ansatze, entanglement entropy, barren plateau diagnostics,
  exact diagonalization benchmarks.
- **B2 — SOC QAOA**: classical MLP surrogate trained on Materials Project spin Hall
  angle data, used as oracle for a QAOA composition optimizer.

## Structure

```
spinq-vqe/
├── src/spinq_vqe/
│   ├── kagome.py        # Lattice + Heisenberg Hamiltonian
│   ├── ansatz.py        # HEA and MERA ansatze
│   ├── vqe.py           # VQE runner + optimizer
│   ├── entanglement.py  # Von Neumann entropy, mutual information
│   ├── utils.py         # Plots (pastel palette)
│   ├── surrogate.py     # MLP surrogate on MP data  [WIP]
│   └── qaoa.py          # QAOA circuit + optimizer  [WIP]
├── notebooks/           # Executable research notebooks
├── figures/             # Generated plots
├── data/                # ED reference energies, MP data
├── OVERVIEW.md          # Full program description
└── REFERENCES.md        # 40+ literature references
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
```

Requires Python ≥ 3.11. Core dependencies: `pennylane`, `jax[cpu]`, `numpy`, `scipy`, `networkx`.

## Status

🔬 **Active research — not yet stable.** Notebooks 01 (Hamiltonian + ED) is complete.
VQE runs, entanglement analysis, surrogate, and QAOA are in progress.

## References

See [`REFERENCES.md`](REFERENCES.md) for the full bibliography.  
Key: Sachdev (1992), Yan/Huse/White (2011), Kandala et al. (2017), Cerezo et al. (2021),
Nakatsuji et al. (2022), UTokyo SOT switching (2026).

---

**License:** MIT &nbsp;·&nbsp; **Contact:** [qondra@arpacorp.net](mailto:qondra@arpacorp.net)
