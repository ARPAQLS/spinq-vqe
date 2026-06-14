# spinq-vqe — References

> Curated references for the `spinq-vqe` repository.
> Covers foundational physics, algorithms, frameworks, datasets, and tools.
> Used to populate the paper bibliography, README, and notebook citations.

---

## Experimental Trigger

| # | Reference | URL |
|---|-----------|-----|
| E1 | Nakatsuji et al. (2026) — SOT switching of Mn₃Sn at 40 ps. *Science* | https://www.science.org/doi/10.1126/science.adt3136 |

---

## Foundational Physics Papers

| # | Reference | URL / DOI |
|---|-----------|-----------|
| P1 | Sachdev, S. (1992) — *Kagome- and triangular-lattice Heisenberg antiferromagnets: Ordering from quantum fluctuations and quantum-disordered ground states with unconfined bosonic spinons.* Phys. Rev. B 45, 12377 | https://doi.org/10.1103/PhysRevB.45.12377 |
| P2 | Yan, S., Huse, D. A., White, S. R. (2011) — *Spin-Liquid Ground State of the S = 1/2 Kagome Heisenberg Antiferromagnet.* Science 332, 1173 | https://doi.org/10.1126/science.1201080 |
| P3 | Depenbrock, S., McCulloch, I. P., Schollwöck, U. (2012) — *Nature of the spin-liquid ground state of the S=1/2 Heisenberg model on the kagome lattice.* Phys. Rev. Lett. 109, 067201 | https://doi.org/10.1103/PhysRevLett.109.067201 |
| P4 | Sinova, J., Valenzuela, S. O., Wunderlich, J., Back, C. H., Jungwirth, T. (2015) — *Spin Hall effects.* Rev. Mod. Phys. 87, 1213 | https://doi.org/10.1103/RevModPhys.87.1213 |
| P5 | Yuasa, S., Nagahama, T., Fukushima, A., Suzuki, Y., Ando, K. (2004) — *Giant room-temperature magnetoresistance in single-crystal Fe/MgO/Fe magnetic tunnel junctions.* Nature Materials 3, 868 | https://doi.org/10.1038/nmat1257 |
| P6 | Nakatsuji, S., Ishizuka, H. (2022) — *Topological and magnetic phases with strong spin-orbit coupling in the Mn₃X family.* Annals of Physics 447, 169146 | https://doi.org/10.1016/j.aop.2022.169146 |
| P7 | Balents, L. (2010) — *Spin liquids in frustrated magnets.* Nature 464, 199 | https://doi.org/10.1038/nature08917 |

---

## Core Algorithm Papers

| # | Reference | URL / DOI |
|---|-----------|-----------|
| A1 | Peruzzo, A. et al. (2014) — *A variational eigenvalue solver on a photonic chip.* Nature Communications 5, 4213 — **Original VQE paper** | https://doi.org/10.1038/ncomms5213 |
| A2 | Farhi, E., Goldstone, J., Gutmann, S. (2014) — *A Quantum Approximate Optimization Algorithm.* arXiv:1411.4028 — **Original QAOA paper** | https://arxiv.org/abs/1411.4028 |
| A3 | Cerezo, M. et al. (2021) — *Variational quantum algorithms.* Nature Reviews Physics 3, 625 — **VQE review** | https://doi.org/10.1038/s42254-021-00348-9 |
| A4 | Tilly, J. et al. (2022) — *The Variational Quantum Eigensolver: A review of methods and best practices.* Physics Reports 986, 1–128 | https://doi.org/10.1016/j.physrep.2022.08.003 |
| A5 | McClean, J. R. et al. (2018) — *Barren plateaus in quantum neural network training landscapes.* Nature Communications 9, 4812 | https://doi.org/10.1038/s41467-018-07090-4 |
| A6 | Wang, S. et al. (2021) — *Noise-induced barren plateaus in variational quantum algorithms.* Nature Communications 12, 6961 | https://doi.org/10.1038/s41467-021-27045-6 |
| A7 | Wecker, D. et al. (2015) — *Progress towards practical quantum advantage in quantum chemistry.* Phys. Rev. A 92, 042303 | https://doi.org/10.1103/PhysRevA.92.042303 |
| A8 | Hadfield, S. et al. (2019) — *From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz.* Algorithms 12, 34 | https://doi.org/10.3390/a12020034 |

---

## VQE for Spin Models (Closest Prior Work)

| # | Reference | URL / DOI |
|---|-----------|-----------|
| S1 | Cade, C. et al. (2020) — *Strategies for solving the Fermi-Hubbard model on near-term quantum computers.* Phys. Rev. B 102, 235122 | https://doi.org/10.1103/PhysRevB.102.235122 |
| S2 | Consiglio, M. et al. (2022) — *Variational quantum eigensolver for the Heisenberg antiferromagnet on a kagome lattice.* Phys. Rev. Research 4, 033257 | https://doi.org/10.1103/PhysRevResearch.4.033257 |
| S3 | Lyu, C. et al. (2023) — *Accelerated variational quantum eigensolver with joint Bell measurement.* PRX Quantum 4, 020337 | https://doi.org/10.1103/PRXQuantum.4.020337 |
| S4 | Seki, K., Shirakawa, T., Yunoki, S. (2020) — *Symmetry-adapted variational quantum eigensolver.* Phys. Rev. A 101, 052340 | https://doi.org/10.1103/PhysRevA.101.052340 |

---

## Quantum Computing Frameworks

| # | Resource | URL |
|---|----------|-----|
| F1 | **PennyLane** — Primary QML/QP framework (Xanadu). Convention: `import pennylane as qp` since March 2026 | https://pennylane.ai |
| F1b | PennyLane blog — *import pennylane as qp: growing beyond just QML* (Josh Izaac, March 19 2026) | https://pennylane.ai/blog/2026/03/import-pennylane-as-qp-growing-beyond-just-qml |
| F2 | PennyLane documentation | https://docs.pennylane.ai |
| F3 | PennyLane `qml.spin` module (Heisenberg, Kagome Hamiltonians) | https://docs.pennylane.ai/en/stable/code/qml_spin.html |
| F4 | PennyLane `qml.qaoa` module | https://docs.pennylane.ai/en/stable/code/qml_qaoa.html |
| F5 | PennyLane demos — VQE | https://pennylane.ai/qml/demos/tutorial_vqe |
| F6 | PennyLane demos — QAOA intro | https://pennylane.ai/qml/demos/tutorial_qaoa_intro |
| F7 | PennyLane demos — Spin Hamiltonians | https://pennylane.ai/qml/demos/tutorial_how_to_build_spin_hamiltonians |
| F8 | **JAX** — Autodiff / JIT backend | https://github.com/google/jax |
| F9 | JAX documentation | https://jax.readthedocs.io |
| F10 | **QuSpin** — Exact diagonalization baseline | https://quspin.github.io/QuSpin |
| F11 | **OpenFermion** — Pauli string / Hamiltonian tools (Google) | https://quantumai.google/openfermion |
| F12 | OpenFermion GitHub | https://github.com/quantumlib/OpenFermion |
| F13 | **NetworkX** — Lattice graph construction | https://networkx.org |
| F14 | **NumPy** | https://numpy.org |
| F15 | **SciPy** — Classical optimizers (COBYLA, L-BFGS-B) | https://scipy.org |
| F16 | **Matplotlib** — Publication figures | https://matplotlib.org |
| F17 | **Plotly** — Interactive notebook plots | https://plotly.com/python |
| F18 | **JupyterLab** | https://jupyterlab.readthedocs.io |

---

## Datasets & APIs

| # | Resource | URL |
|---|----------|-----|
| D1 | **Materials Project** — Crystal structures, DFT properties (AHC, θ_SH, MAE) | https://materialsproject.org |
| D2 | Materials Project API documentation | https://api.materialsproject.org |
| D3 | `mp-api` Python client | https://github.com/materialsproject/api |
| D4 | Materials Project — Mn₃Sn entry | https://materialsproject.org/materials/mp-989807 |

---

## Ansatz & Circuit Design References

| # | Reference | URL / DOI |
|---|-----------|-----------|
| C1 | Kandala, A. et al. (2017) — *Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets.* Nature 549, 242 — **Hardware-efficient ansatz** | https://doi.org/10.1038/nature23879 |
| C2 | Vidal, G. (2007) — *Entanglement Renormalization.* Phys. Rev. Lett. 99, 220405 — **MERA ansatz** | https://doi.org/10.1103/PhysRevLett.99.220405 |
| C3 | Wiersema, R. et al. (2020) — *Exploring entanglement and optimization within the Hamiltonian variational ansatz.* PRX Quantum 1, 020319 | https://doi.org/10.1103/PRXQuantum.1.020319 |

---

## Benchmarking & Classical Comparisons

| # | Reference | URL / DOI |
|---|-----------|-----------|
| B1 | White, S. R. (1992) — *Density matrix formulation for quantum renormalization groups.* Phys. Rev. Lett. 69, 2863 — **DMRG** | https://doi.org/10.1103/PhysRevLett.69.2863 |
| B2 | Weinstein, M. et al. (2023) — *QuSpin: a Python package for dynamics and exact diagonalisation of quantum many-body systems.* SciPost Phys. Codebases 2 | https://doi.org/10.21468/SciPostPhysCodeb.2 |
| B3 | Shende, V. V., Markov, I. L., Bullock, S. S. (2004) — *Minimal universal two-qubit circuits.* Phys. Rev. A 69, 062321 | https://doi.org/10.1103/PhysRevA.69.062321 |

---

## QUBO / Combinatorial Optimization References

| # | Reference | URL / DOI |
|---|-----------|-----------|
| Q1 | Lucas, A. (2014) — *Ising formulations of many NP problems.* Frontiers in Physics 2, 5 — **QUBO encoding reference** | https://doi.org/10.3389/fphy.2014.00005 |
| Q2 | Blekos, K. et al. (2024) — *A review of QAOA: An algorithmic lens on optimization and physics.* Physics Reports 1068, 1–66 | https://doi.org/10.1016/j.physrep.2024.03.002 |
| Q3 | Zhou, L. et al. (2020) — *Quantum Approximate Optimization Algorithm: Performance, mechanism, and implementation on near-term devices.* Phys. Rev. X 10, 021067 | https://doi.org/10.1103/PhysRevX.10.021067 |

---

## arXiv Sections to Monitor

| Section | Scope |
|---------|-------|
| `cond-mat.str-el` | Strongly correlated electrons, frustrated magnets, Kagome AFM |
| `cond-mat.mes-hall` | Spintronics, MTJ, SOT, spin Hall |
| `quant-ph` | VQE, QAOA, variational algorithms, quantum simulation |
| `cond-mat.mtrl-sci` | Materials discovery, DFT, spintronic properties |

---

*Last updated: 2026-06-03 | spinq-vqe / ARPA Spintronics QML Research Program*
